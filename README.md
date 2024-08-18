![Intro Image](assets/themistoai.png)  
## Summary
by TheMisto.ai @Shenzhen, China  

For the Flux-dev model based on the Flow matching structure, a scalable Transformer model is used as the backbone of this ControlNet.  

Structurally, it largely follows @lllyasviel's design philosophy for ControlNet, using zero initialization and gradually introducing the influence of ControlNet.
This CN adopts Dual-stream architecture, consistent with the original design intent of the Flux model, leaving open the possibility for more modalities as CN inputs in the future.  

Inspired by SD3's CN and XLabs’ works, various experiments will be conducted in future open-source projects or models through some scalable modules to test the balance between effectiveness and computational resources. Currently, it may be necessary to optimize part of the qkv to accelerate model running speed.  
This ControlNet is compatible with Flux1.dev's fp16/fp8 and other quantized models.

### Apply with Different Line Preprocessors

### Recommended Parameters


## Models

### Huggingface（抱抱脸）:

### 中国（大陆地区）便捷下载地址:

## Usage

### ComfyUI


## Training Details
The introduction of the Transformer structure and scale law will have a significant impact on training time and computational power. The training cost of MistoLine_Flux1_dev is approximately 2.5 times that of MistoLineSDXL.  
This training was conducted using A100-80GB GPUs with bf16 mixed precision. The image quality is slightly lower than that of MistoLine-SDXL.  
If training on higher resolution datasets is required, distributed training and optimization of the training code will be necessary.


## Models License
Align to the FLUX.1 [dev] Non-Commercial License.  
This ComfyUI node fall under ComfyUI.  
本模型仅供研究和学习，不可用于任何形式商用

## Special thanks
XLabs https://xlabs.by/
@lllyasviel


## Business Cooperation（商务合作）
For any custom model training, commercial cooperation, AI application development, or other business collaboration matters  
Please contact E-mail info@themisto.ai  

如果有任何模型定制训练，商业合作，AI应用开发等商业合作事宜请联系。  
电邮：info@themisto.ai

## WIP
Flux-dev-MistoCN-collection  
Flux-dev-Finetune and SFT

## One more thing


## Our social media
### Global:  
website: https://www.themisto.ai/

### Mainland China:
