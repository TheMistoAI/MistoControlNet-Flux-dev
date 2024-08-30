import torch
import os
import comfy.model_management
from comfy.utils import ProgressBar
import folder_paths
import numpy as np
from safetensors.torch import load_file
from einops import rearrange,repeat
from .modules.misto_controlnet import MistoControlNetFluxDev
from .modules.utils import get_schedule,get_noise,denoise_controlnet, unpack
from PIL import Image,ImageOps

dir_TheMistoModel = os.path.join(folder_paths.models_dir, "TheMisto_model")
os.makedirs(dir_TheMistoModel, exist_ok=True)
folder_paths.folder_names_and_paths["TheMisto_model"] = ([dir_TheMistoModel], folder_paths.supported_pt_extensions)

class LATENT_PROCESSOR_COMFY:
    def __init__(self):
        self.scale_factor = 0.3611
        self.shift_factor = 0.1159
        self.latent_rgb_factors =[
                    [-0.0404,  0.0159,  0.0609],
                    [ 0.0043,  0.0298,  0.0850],
                    [ 0.0328, -0.0749, -0.0503],
                    [-0.0245,  0.0085,  0.0549],
                    [ 0.0966,  0.0894,  0.0530],
                    [ 0.0035,  0.0399,  0.0123],
                    [ 0.0583,  0.1184,  0.1262],
                    [-0.0191, -0.0206, -0.0306],
                    [-0.0324,  0.0055,  0.1001],
                    [ 0.0955,  0.0659, -0.0545],
                    [-0.0504,  0.0231, -0.0013],
                    [ 0.0500, -0.0008, -0.0088],
                    [ 0.0982,  0.0941,  0.0976],
                    [-0.1233, -0.0280, -0.0897],
                    [-0.0005, -0.0530, -0.0020],
                    [-0.1273, -0.0932, -0.0680]
                ]
    def __call__(self, x):
        return (x / self.scale_factor) + self.shift_factor
    def go_back(self, x):
        return (x - self.shift_factor) * self.scale_factor

MAX_RESOLUTION=16384


def prepare_sampling(t5_emb, clip_emb, img,batch_size):
    bs, c, h, w = img.shape
    bs  = batch_size

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if t5_emb.shape[0] == 1 and bs > 1:
        t5_emb = repeat(t5_emb, "1 ... -> bs ...", bs=bs)
    t5_emb_ids = torch.zeros(bs, t5_emb.shape[1], 3)

    if clip_emb.shape[0] == 1 and bs > 1:
        clip_emb = repeat(clip_emb, "1 ... -> bs ...", bs=bs)

    return {
        "img":img,
        "img_ids":img_ids.to(img.device, dtype=img.dtype),
        "txt":t5_emb.to(img.device, dtype=img.dtype),
        "txt_ids":t5_emb_ids.to(img.device, dtype=img.dtype),
        "vec":clip_emb.to(img.device, dtype=img.dtype)
    }


def load_misto_transoformer_cn(device):
    with torch.device(device):
        controlnet = MistoControlNetFluxDev(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            num_heads=24,
            num_transformer=3,
            num_single_transformer=2,
            guidance_embed=True,
        )
    return controlnet


class LoadMistoFluxControlNet:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                             "model_name": (folder_paths.get_filename_list("TheMisto_model"),)
                             }}

    RETURN_TYPES = ("MistoFluxControlNet",)
    RETURN_NAMES = ("ControlNet",)
    FUNCTION = "load_model"
    CATEGORY = "TheMistoAINodes"

    def load_model(self,model_name):
        device=comfy.model_management.get_torch_device()
        misto_cn = load_misto_transoformer_cn(device=device)
        ckpt_path = os.path.join(dir_TheMistoModel, model_name)
        if '.bin' in model_name:
            state_dict = torch.load(ckpt_path, map_location='cpu')
        else:
            state_dict = load_file(ckpt_path)
        miss_, error_ = misto_cn.load_state_dict(state_dict,strict=False)
        misto_cn.eval()
        print(miss_, error_)
        return (misto_cn,)

class ApplyMistoFluxControlNet:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"controlnet": ("MistoFluxControlNet",),
                             "image": ("IMAGE",),
                             "strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.01})
                             }}

    RETURN_TYPES = ("ControlNetCondition",)
    RETURN_NAMES = ("controlnet_condition",)
    FUNCTION = "embedding"
    CATEGORY = "TheMistoAINodes"

    def embedding(self, controlnet, image, strength):
        device=comfy.model_management.get_torch_device()
        cond_img = torch.from_numpy((np.array(image) * 2) - 1)
        cond_img = cond_img.permute(0, 3, 1, 2).to(torch.bfloat16).to(device)
        cond_out = {
            "img": cond_img.to(device),
            "controlnet_strength": strength,
            "model": controlnet,
        }
        return (cond_out,)


class KSamplerTheMisto:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "ae":("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "controlnet_condition": ("ControlNetCondition", {"default": None}),
                "batch_size": ("INT", {"default":1, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.1, "max": 30}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sampling"
    CATEGORY = "TheMistoAINodes"

    def sampling(self, model,ae, positive, negative,controlnet_condition,batch_size,guidance,seed, steps ):
        # device ,dtype and pbar
        device = comfy.model_management.get_torch_device()
        dtype_model = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        pbar = ProgressBar(steps+10)
        pbar.update(1)

        # model
        comfy.model_management.load_model_gpu(model)
        flux_model = model.model.diffusion_model
        pbar.update(3)

        # cn cond
        cn_model = controlnet_condition['model']
        cond_img = controlnet_condition['img']
        cn_strength =  controlnet_condition['controlnet_strength']

        bc, c, h, w = cond_img.shape
        height = (h//16) * 16
        width = (w//16) * 16
        pbar.update(2)
        with torch.no_grad():
            # set scheduler
            timesteps = get_schedule( steps,  (width // 8) * (height // 8) // (16 * 16), shift=True, )
            x = get_noise( 1, height, width, device=device, dtype=dtype_model, seed=seed)
            p_inp_cond = prepare_sampling(positive[0][0], positive[0][1]['pooled_output'], img=x, batch_size=batch_size)
            n_inp_cond = prepare_sampling(negative[0][0], negative[0][1]['pooled_output'], img=x, batch_size=batch_size)
            pbar.update(1)
            # denoise
            x = denoise_controlnet(
                pbar=pbar,
                model=flux_model, **p_inp_cond,
                controlnet=cn_model,
                timesteps=timesteps,
                guidance=guidance,
                controlnet_cond=cond_img,
                controlnet_strength = cn_strength,
                neg_txt=n_inp_cond['txt'],
                neg_txt_ids=n_inp_cond['txt_ids'],
                neg_vec=n_inp_cond['vec'],
            )

            x = unpack(x.float(), height, width)
        lat_processor = LATENT_PROCESSOR_COMFY()
        x = lat_processor(x)
        pbar.update(1)
        return (ae.decode(x),)

NODE_CLASS_MAPPINGS = {
    "LoadTheMistoFluxControlNet": LoadMistoFluxControlNet,
    "ApplyTheMistoFluxControlNet": ApplyMistoFluxControlNet,
    "KSamplerTheMisto":KSamplerTheMisto,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTheMistoFluxControlNet": "Load MistoCN-Flux.dev",
    "ApplyTheMistoFluxControlNet": "Apply MistoCN-Flux.dev",
    "KSamplerTheMisto":"KSampler for MistoCN-Flux.dev",
}