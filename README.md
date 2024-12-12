<br>
<br>
<div align="center">

<picture>
  <img alt="Wings logo" src="https://raw.githubusercontent.com/AIDC-AI/Wings/main/assets/images/logo.png" width="550px">
</picture>
</br>
</br>

</div>

<div id="top"></div>  

<div align="center">
  <h3 align="center">Wings: A Versatile Multimodal LLM without Text-only Forgetting</h3>
</div>
<p align="center">
📝 <a href="https://arxiv.org/abs/2406.03496" target="_blank">Paper</a> • 🤗 <a href="https://huggingface.co/AIDC-AI/Wings-Qwen1_5-8B" target="_blank">Hugging Face</a>
</a>
</p> 

<p align="center">
    🚀 Ask questions or discuss ideas on <a href="https://github.com/AIDC-AI/Wings/issues" target="_blank"> GitHub </a>
</p>

<details>
<summary></b>Table of Contents</b></summary>

- [Why Wings?](#why-wings)
- [How to use](#how-to-use)
  - [Quick start](#quick-start)
  - [Citation](#citation)
- [License](#license)
- [Disclaimer](#disclaimer)
</details>

<br>

## Why Wings?

💡 TL;DR

- Wings is a brand-new universal Multimodal Large Language Model (MLLM). Its flexible multimodal structure enhances the MLLM as if **giving it wings that enhance the performance of multimodal capabilities** while minimizing text-only forgetting.

- **Any** architecture of MLLM can adapt the Wings component.

Multimodal large language models (MLLMs), initiated with a trained LLM, first align images with text and then fine-tune on multimodal mixed inputs. However, the MLLM **catastrophically forgets the text-only instructions**, which do not include images and can be addressed within the initial LLM.

In this work, we present Wings, a novel MLLM that excels in both text-only dialogues and multimodal comprehension. Analyzing MLLM attention in multimodal instructions reveals that **text-only forgetting is related to the attention shifts from pre-image to post-image text.** From that, we construct extra modules that act as the boosted learner to compensate for the attention shift. The complementary visual and textual learners, **like "wings" on either side, are connected in parallel within each layer's attention block.** Initially, image and text inputs are aligned with visual learners operating alongside the main attention, balancing focus on visual elements. Textual learners are later collaboratively integrated with attention-based routing to blend the outputs of the visual and textual learners. We design the **Low-Rank Residual Attention (LoRRA)** to guarantee high efficiency for learners.

Our experimental results demonstrate that Wings outperforms equally-scaled MLLMs in both text-only and visual question-answering tasks. On a newly constructed Interleaved Image-Text (IIT) benchmark, Wings exhibits superior performance from text-only-rich to multimodal-rich question-answering tasks.

<div align="center">

<picture>
  <img alt="Wings logo" src="https://raw.githubusercontent.com/AIDC-AI/Wings/main/assets/images/bench_example.png" width="800px">
</picture>
</br>

</div>

## How to use

### Quick start

+ Environment Setups:
  
  ```python
  conda create --name your_env_name python=3.10
  conda activate your_env_name
  pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt
  ```

+ Training:
  
  ```python
  bash run/pretrain_base.sh
  # Set path for pretrained MLLM (after stage 1)
  bash run/finetune_base.sh
  ```

+ Inferring
  
  ```python
  # Set path containing the trained safetensors
  bash run/infer.sh
  ```

### Citation

+ If you find Wings useful, please cite the paper:
  
  ```
  @article{zhang_wings,
    author       = {Yi{-}Kai Zhang and
                    Shiyin Lu and
                    Yang Li and
                    Yanqing Ma and
                    Qing{-}Guo Chen and
                    Zhao Xu and
                    Weihua Luo and
                    Kaifu Zhang and
                    De{-}Chuan Zhan and
                    Han{-}Jia Ye},
    title        = {Wings: Learning Multimodal LLMs without Text-only Forgetting},
    journal      = {CoRR},
    volume       = {abs/2406.03496},
    year         = {2024}
  }
  ```

## License

This project is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt) (SPDX-License-Identifier: Apache-2.0).

## Disclaimer

We used compliance-checking algorithms during the training process, to ensure the compliance of the trained model to the best of our ability. Due to the complexity of the data and the diversity of language model usage scenarios, we cannot guarantee that the model is completely free of copyright issues or improper content. If you believe anything infringes on your rights or generates improper content, please contact us, and we will promptly address the matter.

## Acknowledgements

We express our gratitude to the [LLaVA](https://github.com/haotian-liu/LLaVA) project and its contributors. Special thanks to [Xudong Lu](https://github.com/Lucky-Lance) for fixing the bug in `modeling_wings_llama.py`.
