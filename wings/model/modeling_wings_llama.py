#    Copyright (C) 2024 AIDC-AI
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#        http://www.apache.org/licenses/LICENSE-2.0
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.cache_utils import Cache, DynamicCache
# from transformers.cache_utils import Cache, DynamicCache, StaticCache

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

import transformers
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM, AutoConfig, AutoModelForCausalLM
from transformers.utils.import_utils import is_torchdynamo_compiling

from wings.model.base_architecture import LlavaMetaModel, WingsMetaForCausalLM
from wings.model.base_module import ReweightLinear, WingsAttention

from wings.utils import logger

def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask

class WingsLlamaConfig(LlamaConfig):
    model_type = "wings_llama"
    def __init__(self, **kwargs):
        super(WingsLlamaConfig, self).__init__(**kwargs)

class WingsLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = WingsLlamaConfig

    def __init__(self, config):
        super(WingsLlamaModel, self).__init__(config)

def wings_layer_forward(self):
    def forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_value: Optional[Cache],
        output_attentions: Optional[bool],
        use_cache: Optional[bool],
        cache_position: Optional[torch.LongTensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        image_features: Optional[torch.Tensor],
        text_features: Optional[torch.Tensor],
        position_ids_image: Optional[torch.Tensor],
        position_ids_text: Optional[torch.Tensor],
        padding_length: Optional[List] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        hidden_states_wings = []
        if image_features is not None and not isinstance(image_features, list) and position_ids_image is not None and len(position_ids_image) != 0:
            hidden_states_image, _, _ = self.attn_pool(
                query=hidden_states,
                key=image_features,
                value=image_features,
                attention_mask=None,
                position_ids_q=position_ids,
                position_ids_image=position_ids_image,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache
            )
            hidden_states_wings.append(hidden_states_image)

        if hasattr(self, 'reweight_module') and text_features is not None:
            hidden_states_text, _, _ = self.attn_t_pool(
                query=hidden_states,
                key=text_features,
                value=text_features,
                attention_mask=attention_mask,
                position_ids_q=position_ids,
                position_ids_image=position_ids_text,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_length=padding_length
            )
            hidden_states_wings.append(hidden_states_text)

        if len(hidden_states_wings) > 1:
            reweight = self.reweight_module(self_attn_weights)
            if padding_length is not None:
                if all([i == 0 for i in padding_length]):
                    pad_mask2image = torch.zeros_like(reweight[0])
                else:
                    pad_mask2image = torch.ones_like(reweight[0])
                    pad_mask2image[[pad_mask_i for pad_mask_i, pad_mask_v in enumerate(padding_length) if pad_mask_v == 0], :, 0] = 0
                hidden_states = hidden_states + reweight[0] * hidden_states_wings[0] * pad_mask2image
            else:
                hidden_states = hidden_states + reweight[0] * hidden_states_wings[0]
            hidden_states = hidden_states + reweight[1] * hidden_states_wings[1]

            for reweight_i, hidden_state_i in zip(reweight, hidden_states_wings):
                hidden_states = hidden_states + reweight_i * hidden_state_i
        elif len(hidden_states_wings) == 1:
            hidden_states = hidden_states + hidden_states_wings[0]

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    return forward

def wings_forward(self):
    def forward(
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        position_ids_image: Optional[torch.Tensor] = None,
        position_ids_text: Optional[torch.Tensor] = None,
        use_cache_for_image: Optional[bool] = False,
        output_wings_loss: Optional[bool] = False,
        padding_length: Optional[List] = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if (
            use_cache and not isinstance(past_key_values, Cache) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_wings_loss = [] if output_wings_loss else None
        next_decoder_cache = None

        for cur_layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    image_features,
                    text_features,
                    position_ids_image,
                    position_ids_text,
                    padding_length
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    image_features=image_features,
                    text_features=text_features,
                    position_ids_image=position_ids_image,
                    position_ids_text=position_ids_text,
                    padding_length=padding_length
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_wings_loss:
                all_wings_loss.extend(layer_outputs[-1])

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    return forward

class WingsLlamaForCausalLM(LlamaForCausalLM, WingsMetaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    config_class = WingsLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = WingsLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_model(self):
        return self.model

    def initialize_modules(self, model_args, data_args, training_args):
        self.config.lora_dim = model_args.lora_dim
        self.config.model_max_length = model_args.model_max_length
        self.config.system_prompt_length = model_args.system_prompt_length

        self.config.image_aspect_ratio = data_args.image_aspect_ratio
        self.config.image_token_length = data_args.image_token_length

        self.config.use_cache = training_args.use_cache
        self.config.tune_only_mm_mlp_adapter = training_args.tune_only_mm_mlp_adapter
        self.config.mm_projector_lr = training_args.mm_projector_lr
        self.config.vision_tower_lr_follow_mm_projector = training_args.vision_tower_lr_follow_mm_projector
        self.config.lr_projector_follow_tuned_keys = training_args.lr_projector_follow_tuned_keys

        for cur_layer_index in model_args.attn_layers_idx:
            self.model.layers[cur_layer_index].attn_pool = WingsAttention(self.config, cur_layer_index).to(torch.bfloat16)
            self.model.layers[cur_layer_index].attn_t_pool = WingsAttention(self.config, cur_layer_index).to(torch.bfloat16)
            if model_args.wings_router_type == 'linear':
                # TODO: find the hidden_size (4096)
                self.model.layers[cur_layer_index].reweight_module = ReweightLinear(4096).to(torch.bfloat16)

        for m in self.model.layers:
            m.forward = wings_layer_forward(m)

        self.model.forward = wings_forward(self.model)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        images: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, i_t_indices, image_features
            ) = self.prepare_multimodal_inputs(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, get_image_features=True
            )
            text_features = inputs_embeds

        image_begin_end = []
        if i_t_indices is not None:
            for i_t_indices_pair in i_t_indices:
                base_index = 0
                for slice_idx, slice in enumerate(i_t_indices_pair):
                    if slice[1] != -1:
                        if slice[0] == 'i':
                            image_begin_end.append((base_index, base_index + self.config.image_token_length))
                            break
                        else:
                            base_index += slice[1]
                    else:
                        image_begin_end.append((0, self.config.image_token_length))

            assert len(i_t_indices) == len(image_begin_end)
        position_ids_image, position_ids_text = [], []
        padding_length = None
        if len(image_begin_end) > 0:
            position_ids_image.extend([
                torch.arange(
                    image_token_begin, image_token_end, dtype=torch.long, device=inputs_embeds.device
                ) for image_token_begin, image_token_end in image_begin_end
            ])
            position_ids_image = torch.stack(position_ids_image, dim=0)

            for image_token_begin, image_token_end in image_begin_end:
                temp_pos = torch.arange(0, inputs_embeds.shape[1])
                if image_token_begin == 0:
                    position_ids_text.append(temp_pos)
                else:
                    position_ids_text.append(torch.cat(
                        (temp_pos[self.config.system_prompt_length:image_token_begin], temp_pos[image_token_end:]), dim=0
                    ))
            max_length = max([len(i_l) for i_l in position_ids_text])
            is_padding = any([len(i_l) != max_length for i_l in position_ids_text])
            if is_padding:
                padding_length = [max_length - len(i) for i in position_ids_text]
                position_ids_text = [i if len(i) == max_length else F.pad(i, (0, pad_l), mode='constant', value=i[-1]) for pad_l, i in zip(padding_length, position_ids_text)]
            text_features = torch.stack([text_features[batch_idx, cur_t_indices, :] for batch_idx, cur_t_indices in enumerate(position_ids_text)])
            position_ids_text = torch.stack(position_ids_text, dim=0)
        else:
            position_ids_text = torch.arange(self.config.system_prompt_length, inputs_embeds.shape[1]).unsqueeze(0).repeat(inputs_embeds.shape[0], 1)
            text_features = torch.stack([text_features[batch_idx, cur_t_indices, :] for batch_idx, cur_t_indices in enumerate(position_ids_text)])

        if not isinstance(image_features, list) and inputs_embeds.shape[1] < image_features.shape[1]:
            image_features = image_features[:, :inputs_embeds.shape[1], :]
            padding_length = [0 for _ in range(image_features.shape[0])]
            if len(position_ids_image) != 0 and inputs_embeds.shape[1] < position_ids_image.shape[1]:
                position_ids_image = position_ids_image[:, :inputs_embeds.shape[1]]

        position_ids_text = position_ids_text.to(dtype=torch.long, device=inputs_embeds.device)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            image_features=image_features,
            text_features=text_features,
            position_ids_image=position_ids_image,
            position_ids_text=position_ids_text,
            padding_length=padding_length
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            if labels is None and not is_torchdynamo_compiling():
                logger.warning_once(
                    "Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)"
                )
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None)
            }
        )
        return model_inputs

    @classmethod
    def build(cls, model_name, model_path, **kwargs):
        model_kwargs = {k: v for k, v in kwargs.items() if k in cls.MODEL_BUILD_KEYS}

        model = cls.from_pretrained(
            model_path,
            model_type=WingsLlamaConfig.model_type,
            attn_implementation="eager",
            **model_kwargs
        )

        tokenizer_kwargs = {k: v for k, v in kwargs.items() if k in cls.TOKENIZER_BUILD_KEYS}
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            **tokenizer_kwargs
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return model, tokenizer

AutoConfig.register("wings_llama", WingsLlamaConfig)
AutoModelForCausalLM.register(WingsLlamaConfig, WingsLlamaForCausalLM)
