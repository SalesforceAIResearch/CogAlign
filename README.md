# [ACL 2025 Findings] Why Vision Language Models Struggle with Visual Arithmetic? Towards Enhanced Chart and Geometry Understanding



<div align="center">
<a href="https://khuangaf.github.io/">Kung-Hsiang Huang</a>, Can Qin, Haoyi Qiu, Philippe Laban, Shafiq Joty, Caiming Xiong, Chien-Sheng Wu

</div>
<div align="center">
<strong>Salesforce AI Research</strong>
</div>

<hr>

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2312.10160-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2312.10160) -->

<a href='https://arxiv.org/abs/2502.11492'><img src='https://img.shields.io/badge/arXiv-2502.11492-b31b1b.svg'></a>
[![CogAlign](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CogAlign-blue)](https://huggingface.co/datasets/Salesforce/CogAlign) 
[![CogAlignLLaVAOV0.5B](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-cogalign_llava_ov_0_5b-blue)](https://huggingface.co/Salesforce/cogalign-llava-ov-0_5b)
[![CogAlignLLaVAOV0.5B](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-cogalign_internvl2.5_mpo_1b-blue)](https://huggingface.co/Salesforce/cogalign-internvl2_5-mpo-1b)
[![CogAlignLLaVAOV0.5B](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-cogalign_internvl2.5_mpo_4b-blue)](https://huggingface.co/Salesforce/cogalign-internvl2_5-mpo-4b)
<a href='https://github.com/SalesforceAIResearch/CogAlign/blob/main/LICENSE.txt'><img src='https://img.shields.io/badge/License-CC_NC_4.0-blue'></a>
## Abstract

Vision Language Models (VLMs) have achieved remarkable progress in multimodal tasks, yet they often struggle with visual arithmetic, seemingly simple capabilities like object counting or length comparison, which are essential for relevant complex tasks like chart understanding and geometric reasoning. In this work, we first investigate the root causes of this deficiency through a suite of probing tasks focusing on basic visual arithmetic. Our analysis reveals that while pre-trained vision encoders typically capture sufficient information, the text decoder often fails to decode it correctly for arithmetic reasoning. To address this, we propose CogAlign, a novel post-training strategy inspired by Piaget's theory of cognitive development. CogAlign trains VLMs to recognize invariant properties under visual transformations. We demonstrate that this approach significantly improves the performance of three diverse VLMs on our proposed probing tasks. Furthermore, CogAlign enhances performance by an average of 4.6% on CHOCOLATE and 2.9% on MATH-VISION, outperforming or matching supervised fine-tuning methods while requiring only 60% less training data. These results highlight the effectiveness and generalizability of CogAlign in improving fundamental visual arithmetic capabilities and their transfer to downstream tasks.

Note: This repository is for research purposes only and not for commerical. This dataset introduced was generated using gpt-4o and should not be used to develop models that compete with OpenAI.

## Table of Contents

- [Accessing the Data](#accessing-the-data)
- [Downloading Models](#downloading-models)
- [Running Inference](#running-inference)
- [Citation](#citation)
- [Ethical Considerations](#ethical-considerations)


## Accessing the Data

We publicly release the 64K training data for CogAlign on [🤗 Huggingface](https://huggingface.co/datasets/Salesforce/CogAlign).

```python
from datasets import load_dataset

dataset = load_dataset("Salesforce/CogAlign")['train']
```

## Downloading Models 

### CogAlign-InternVL 

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "Salesforce/cogalign-internvl2_5-mpo-1b"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
```

### CogAlign-LLaVA-OV

You will need to install LLaVA-Next first:

```bash
pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git`
```

```python
from llava.model.builder import load_pretrained_model
tokenizer, model, image_processor, _ = load_pretrained_model(
    "Salesforce/cogalign-llava-ov-0_5b",
    None,
    "llava_qwen",
    device_map="cpu",
)
model.cuda()
```
### Running Inference

Please refer to the `model_inference.ipynb` notebook.

### Citation

If you find this work useful, please consider citing:

```bibtex

@inproceedings{huang-etal-2025-cogalign,
title = "Why Vision Language Models Struggle with Visual Arithmetic? Towards Enhanced Chart and Geometry Understanding",
author = "Huang, Kung-Hsiang  and
  Qin, Can  and
  Qiu, Haoyi  and
  Laban, Philippe  and
  Joty, Shafiq  and
  Xiong, Caiming  and
  Wu, Chien-Sheng",
year = "2025",
booktitle = "Findings of the Association for Computational Linguistics: ACL 2025"
}

```

## Ethical Considerations
This release is for research purposes only in support of an academic paper. Our models, datasets, and code are not specifically designed or evaluated for all downstream purposes. We strongly recommend users evaluate and address potential concerns related to accuracy, safety, and fairness before deploying this model. We encourage users to consider the common limitations of AI, comply with applicable laws, and leverage best practices when selecting use cases, particularly for high-risk scenarios where errors or misuse could significantly impact people’s lives, rights, or safety. For further guidance on use cases, refer to our AUP and AI AUP. 
