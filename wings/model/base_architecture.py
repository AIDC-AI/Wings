from abc import ABC, abstractmethod
from importlib import import_module

import torch
import torch.nn as nn

from wings.model.multimodal_encoder.base import build_vision_tower
from wings.model.multimodal_projector.base import build_vision_projector
from wings.model.conversation_formatter import ConversationFormatter

from wings.configs import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from wings.utils import logger


class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=False)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class WingsMetaForCausalLM(ABC):
    MODEL_MAPPING = {
        "llava_qwen2": "LLaVAQwen2ForCausalLM",
        "wings_qwen2": "WingsQwen2ForCausalLM",
        "wings_llama3_1": "WingsLlamaForCausalLM"
    }
    MODEL_BUILD_KEYS = ['mm_vision_tower', 'low_cpu_mem_usage', 'torch_dtype', 'device_map', 'attn_implementation']
    TOKENIZER_BUILD_KEYS = ['model_max_length']

    @classmethod
    def build(cls, model_name, model_path, conversation_formatter_kwargs, **kwargs):
        model_cls = getattr(import_module(__package__), cls.MODEL_MAPPING[model_name])
        model, tokenizer = model_cls.build(model_name, model_path, **kwargs)
        conversation_formatter = ConversationFormatter(tokenizer, **conversation_formatter_kwargs)
        return model, tokenizer, conversation_formatter

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)

        return image_features

    def prepare_multimodal_inputs(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, get_image_features=False
    ):
        if input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None

        image_features = []
        if images is not None:
            if type(images) is list or images.ndim == 5:
                if type(images) is list:
                    images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
                concat_images = torch.cat([image for image in images], dim=0)
                if concat_images.dtype != torch.bfloat16:
                    concat_images = concat_images.to(torch.bfloat16)
                source_features = self.encode_images(concat_images)
                split_sizes = [image.shape[0] for image in images]
                source_features = torch.split(source_features, split_sizes, dim=0)
                mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
                image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
                if mm_patch_merge_type == 'flat':
                    source_features = [x.flatten(0, 1) for x in source_features]
                elif mm_patch_merge_type.startswith('spatial'):
                    for image_idx, image_feature in enumerate(source_features):
                        if image_feature.shape[0] > 1:
                            base_image_feature = image_feature[0]
                            image_feature = image_feature[1:]
                            height = width = self.get_vision_tower().num_patches_per_side
                            assert height * width == base_image_feature.shape[0]
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(
                                        image_feature.device)
                                ), dim=-1)
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            else:
                                image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                                image_feature = image_feature.flatten(0, 3)
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        else:
                            image_feature = image_feature[0]
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                        image_features.append(image_feature)
                else:
                    raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
            else:
                images = images.to(torch.bfloat16) if images.dtype != torch.bfloat16 else images
                image_features = self.encode_images(images)
                # logger.info(f"Encoding the image: from {images.shape} to {image_features.shape}")


        if getattr(self.config, 'tune_only_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        labels_init = labels
        position_ids_init = position_ids
        attention_mask_init = attention_mask
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        num_virtual_tokens = attention_mask.shape[1] - input_ids.shape[1] if input_ids.shape[1] < attention_mask.shape[1] else 0
        input_ids = [cur_input_ids[cur_attention_mask[num_virtual_tokens:]] for cur_input_ids, cur_attention_mask in
                     zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask[num_virtual_tokens:]] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        input_i_t_ids_list, input_embeddings_list, labels_list = [], [], []
        image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if len(image_features) == 0 or num_images == 0:
                input_embeddings = torch.cat([self.get_model().embed_tokens(cur_input_ids)], dim=0)
                input_embeddings_list.append(input_embeddings)
                labels_list.append(labels[batch_idx])
                image_idx += 1
                input_i_t_ids_list.append([('t', input_embeddings.shape[0]), ('i', -1)])
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            input_ids_noimg, labels_noimg = [], []
            cur_labels = labels[batch_idx]
            for i in range(len(image_token_indices) - 1):
                input_ids_noimg.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                labels_noimg.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in labels_noimg]
            input_embeddings = self.get_model().embed_tokens(torch.cat(input_ids_noimg))
            input_embeddings_noimg = torch.split(input_embeddings, split_sizes, dim=0)

            input_embeddings_updated, input_i_t_ids_updated, labels_updated = [], [], []

            for i in range(num_images + 1):
                input_embeddings_updated.append(input_embeddings_noimg[i])
                input_i_t_ids_updated.append(('t', len(input_embeddings_noimg[i])))
                labels_updated.append(labels_noimg[i])
                if i < num_images and image_idx < len(image_features):
                    cur_image_features = image_features[image_idx]
                    image_idx += 1
                    input_embeddings_updated.append(cur_image_features)
                    input_i_t_ids_updated.append(('i', len(cur_image_features)))
                    labels_updated.append(
                        torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device,
                                   dtype=cur_labels.dtype))

            input_embeddings_updated = [x.to(self.device) for x in input_embeddings_updated]

            input_embeddings_updated = torch.cat(input_embeddings_updated)
            labels_updated = torch.cat(labels_updated)
            # logger.info(f"Modified input emb: from {input_embeddings.shape} to {input_embeddings_updated.shape}")

            input_embeddings_list.append(input_embeddings_updated)
            input_i_t_ids_list.append(input_i_t_ids_updated)
            labels_list.append(labels_updated)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_max_length = getattr(self.config, 'tokenizer_max_length', None)
        if tokenizer_max_length is not None:
            input_embeddings_list = [x[:tokenizer_max_length] for x in input_embeddings_list]
            labels_list = [x[:tokenizer_max_length] for x in labels_list]

        # Combine them
        max_len = max(x.shape[0] for x in input_embeddings_list)
        batch_size = len(input_embeddings_list)

        input_embeddings_padded = []
        labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=labels_list[0].dtype,
                                       device=labels_list[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for idx, (cur_input_embeddings, cur_labels) in enumerate(zip(input_embeddings_list, labels_list)):
            cur_len = cur_input_embeddings.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                input_embeddings_padded.append(
                    torch.cat((
                        torch.zeros(
                            (max_len - cur_len, cur_input_embeddings.shape[1]),
                            dtype=cur_input_embeddings.dtype,
                            device=cur_input_embeddings.device
                        ),
                        cur_input_embeddings),
                    dim=0)
                )
                if cur_len > 0:
                    labels_padded[idx, -cur_len:] = cur_labels
                    attention_mask[idx, -cur_len:] = True
                    position_ids[idx, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                input_embeddings_padded.append(
                    torch.cat((
                        cur_input_embeddings,
                        torch.zeros((max_len - cur_len, cur_input_embeddings.shape[1]), dtype=cur_input_embeddings.dtype,
                                    device=cur_input_embeddings.device)),
                        dim=0)
                    )
                if cur_len > 0:
                    labels_padded[idx, :cur_len] = cur_labels
                    attention_mask[idx, :cur_len] = True
                    position_ids[idx, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        input_embeddings_list = torch.stack(input_embeddings_padded, dim=0)

        labels_list = None if labels_init is None else labels_padded
        attention_mask = None if attention_mask_init is None else attention_mask.to(dtype=attention_mask_init.dtype)
        position_ids = None if position_ids_init is None else position_ids

        if num_virtual_tokens != 0:
            prefix_attention_mask = torch.ones(attention_mask.shape[0], num_virtual_tokens, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        if get_image_features:
            return None, position_ids, attention_mask, past_key_values, input_embeddings_list, labels_list, input_i_t_ids_list, image_features
        return None, position_ids, attention_mask, past_key_values, input_embeddings_list, labels_list, None, None

    def prepare_base_multimodal_inputs(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            if concat_images.dtype != torch.bfloat16:
                concat_images = concat_images.to(torch.bfloat16)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx],
                                                                                            self.config.image_grid_pinpoints,
                                                                                            self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(
                                    image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            if images.dtype != torch.bfloat16:
                images = images.to(torch.bfloat16)
            image_features = self.encode_images(images)
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        num_virtual_tokens = attention_mask.shape[1] - input_ids.shape[1] if input_ids.shape[1] < attention_mask.shape[1] else 0
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask[num_virtual_tokens:]] for cur_input_ids, cur_attention_mask in
                     zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask[num_virtual_tokens:]] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_i_t_indices = []
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                if cur_image_idx < len(image_features):
                    cur_image_features = image_features[cur_image_idx]
                    cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                    cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                else:
                    cur_input_embeds = torch.cat([self.get_model().embed_tokens(cur_input_ids)], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                # noim: no image
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

            cur_new_input_embeds = []
            cur_new_input_i_t_indices = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_input_i_t_indices.append(('t', len(cur_input_embeds_no_im[i])))
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images and cur_image_idx < len(image_features):
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_i_t_indices.append(('i', len(cur_image_features)))
                    cur_new_labels.append(
                        torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device,
                                   dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_input_i_t_indices.append(cur_new_input_i_t_indices)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype,
                                       device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                              device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                             device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        if num_virtual_tokens != 0:
            prefix_attention_mask = torch.ones(attention_mask.shape[0], num_virtual_tokens, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
