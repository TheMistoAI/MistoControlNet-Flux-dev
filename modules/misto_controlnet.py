import torch
from diffusers.utils import is_torch_version
from torch import Tensor, nn
from einops import rearrange
from typing import Any, Dict, Tuple, Union
from .utils import EmbedND, MLPEmbedder, DoubleStreamBlock, SingleStreamBlock, timestep_embedding

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class CondDownsamplBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(16, 16, 1),
            nn.SiLU(),
            zero_module(nn.Conv2d(16, 16, 3, padding=1))
        )

    def forward(self, x):
        return self.encoder(x)


class EnhanceControlnet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.act(self.linear(x))



class MistoControlNetFluxDev(nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(
            self,
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            num_heads=24,
            num_transformer=2,
            num_single_transformer=2,
            guidance_embed=True,
    ):
        super().__init__()
        self.out_channels = in_channels
        self.axes_dim = [16, 56, 56]
        self.theta=10_000
        self.guidance_embed = guidance_embed

        if hidden_size % num_heads != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}")

        pe_dim = hidden_size // num_heads

        if sum(self.axes_dim) != pe_dim:
            raise ValueError(f"Got {self.axes_dim} but expected positional dim {pe_dim}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.pe_embedder = EmbedND(dim=pe_dim, theta=self.theta, axes_dim=self.axes_dim)
        self.img_in = nn.Linear(in_channels, self.hidden_size, bias=True)

        self.txt_in = nn.Linear(context_in_dim, self.hidden_size)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)

        self.vector_in = MLPEmbedder(vec_in_dim, self.hidden_size)
        self.guidance_in = (MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if guidance_embed else nn.Identity())
        self.pos_embed_input = nn.Linear(in_channels, self.hidden_size, bias=True)
        self.gradient_checkpointing = False
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                )
                for _ in range(num_transformer)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=4.0)
                for _ in range(num_single_transformer)
            ]
        )

        # ControlNet blocks
        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(num_transformer):
            controlnet_block = EnhanceControlnet(self.hidden_size)
            controlnet_block = zero_module(controlnet_block)
            self.controlnet_blocks.append(controlnet_block)

        # single controlnet blocks
        self.single_controlnet_blocks = nn.ModuleList([])
        for _ in range(num_single_transformer):
            controlnet_block = EnhanceControlnet(self.hidden_size)
            controlnet_block = zero_module(controlnet_block)
            self.single_controlnet_blocks.append(controlnet_block)

        # Input processing
        self.input_cond_block = CondDownsamplBlock()

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value


    @property
    def attn_processors(self):
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        controlnet_cond: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        controlnet_cond = self.input_cond_block(controlnet_cond)
        controlnet_cond = rearrange(controlnet_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        controlnet_cond = self.pos_embed_input(controlnet_cond)

        img = img + controlnet_cond

        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        block_res_samples = ()
        for block in self.double_blocks:
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    img,
                    txt,
                    vec,
                    pe,
                )
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
            block_res_samples = block_res_samples + (img,)

        img = torch.cat((txt, torch.zeros_like(img)), 1)
        single_block_res_samples = ()
        for index, block in enumerate(self.single_blocks):
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    img,
                    vec,
                    pe,
                )
            else:
                img = block(img, vec=vec, pe=pe)
            single_block_res_samples = single_block_res_samples+(img,)

        controlnet_block_res_samples = ()
        for block_res_sample, controlnet_block in zip(block_res_samples, self.controlnet_blocks):
            block_res_sample = controlnet_block(block_res_sample)
            controlnet_block_res_samples = controlnet_block_res_samples + (block_res_sample,)

        single_controlnet_block_res_samples = ()
        for single_block_res_sample, single_controlnet_block in zip(single_block_res_samples, self.single_controlnet_blocks):
            single_block_res_sample = single_controlnet_block(single_block_res_sample)
            single_controlnet_block_res_samples = single_controlnet_block_res_samples + (single_block_res_sample,)

        return controlnet_block_res_samples,single_controlnet_block_res_samples