"""Inference-only one-step generator used by :mod:`infer`.

The LoRA wrappers live in this file intentionally.  This keeps the inference
folder independent from the training entry points in ``model_training``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline
from PIL import Image


class LinearFP8Wrapper(torch.nn.Module):
    """Run a linear layer with FP8 weights cast to the input dtype on demand."""

    def __init__(self, linear: torch.nn.Linear) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


class SingleLoRALinear(torch.nn.Module):
    """Single-LoRA wrapper matching the module layout used during training."""

    def __init__(self, linear: torch.nn.Linear, rank: int, alpha: int) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.linear = linear
        self.lora_A = torch.nn.Linear(
            linear.in_features, rank, bias=False, device=linear.weight.device, dtype=torch.bfloat16
        )
        self.lora_B = torch.nn.Linear(
            rank, linear.out_features, bias=False, device=linear.weight.device, dtype=torch.bfloat16
        )
        self.scaling = alpha / max(1, rank)
        torch.nn.init.normal_(self.lora_A.weight, std=1.0 / rank)
        torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.linear.weight.detach().to(dtype=x.dtype)
        bias = self.linear.bias
        if bias is not None:
            bias = bias.detach().to(dtype=x.dtype)
        base = F.linear(x, weight, bias)
        return base + self.lora_B(self.lora_A(x)) * self.scaling

    def merge(self) -> torch.nn.Linear:
        """Merge LoRA into a BF16 linear layer for inference."""
        source = self.linear
        merged = torch.nn.Linear(
            source.in_features,
            source.out_features,
            bias=source.bias is not None,
            device=source.weight.device,
            dtype=torch.bfloat16,
        )
        base_weight = source.weight.to(dtype=self.lora_A.weight.dtype)
        update = self.lora_B.weight @ self.lora_A.weight * self.scaling
        merged.weight.data.copy_((base_weight + update).to(torch.bfloat16))
        if source.bias is not None:
            merged.bias.data.copy_(source.bias.to(torch.bfloat16))
        merged.requires_grad_(False)
        return merged


def replace_linear_with_single_lora(
    model: torch.nn.Module,
    patterns: Sequence[str],
    rank: int,
    alpha: int,
    use_fp8: bool,
) -> None:
    """Install the same LoRA/FP8 module structure used by the trainer."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def replace(parent: torch.nn.Module, prefix: str = "") -> None:
        for name, module in list(parent.named_children()):
            full_name = f"{prefix}{name}"
            if not isinstance(module, torch.nn.Linear):
                replace(module, full_name + ".")
                continue

            if use_fp8:
                module.weight.data = module.weight.data.to(torch.float8_e4m3fn)
                if module.bias is not None:
                    module.bias.data = module.bias.data.to(torch.float8_e4m3fn)
            replacement: torch.nn.Module
            if any(pattern in full_name for pattern in patterns):
                replacement = SingleLoRALinear(module, rank, alpha)
            else:
                replacement = LinearFP8Wrapper(module)
            setattr(parent, name, replacement)

    replace(model)


def discover_model_weights(transformer_root: Path, component_root: Path) -> list[str | list[str]]:
    """Discover transformer, text encoder and VAE safetensors under model roots."""
    transformer_files = sorted((transformer_root / "transformer").glob("diffusion_pytorch_model*.safetensors"))
    text_encoder_files = sorted((component_root / "text_encoder").glob("model*.safetensors"))
    vae_file = component_root / "vae" / "diffusion_pytorch_model.safetensors"

    if not transformer_files:
        raise FileNotFoundError(f"No transformer safetensors found under {transformer_root / 'transformer'}")
    if not text_encoder_files:
        raise FileNotFoundError(f"No text encoder safetensors found under {component_root / 'text_encoder'}")
    if not vae_file.is_file():
        raise FileNotFoundError(f"VAE weight not found: {vae_file}")

    return [
        [str(path) for path in transformer_files],
        [str(path) for path in text_encoder_files],
        str(vae_file),
    ]


class Generator(torch.nn.Module):
    """Qwen-Image Edit generator reduced to the trained one-step inference path."""

    LORA_PATTERNS = (
        "img_in",
        "img_mod.1",
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "to_out.0",
        "img_mlp.net.0.proj",
        "img_mlp.net.2",
    )

    def __init__(
        self,
        transformer_root: str | Path,
        component_root: str | Path,
        checkpoint: str | Path,
        *,
        torch_dtype: torch.dtype = torch.bfloat16,
        lora_rank: int = 128,
        use_fp8: bool = True,
        start_timestep: int = 750,
    ) -> None:
        super().__init__()
        transformer_root = Path(transformer_root).expanduser()
        component_root = Path(component_root).expanduser()
        checkpoint = Path(checkpoint).expanduser()
        tokenizer_path = component_root / "tokenizer"
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_path}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        weights = discover_model_weights(transformer_root, component_root)
        model_configs = [ModelConfig(path=path) for path in weights]
        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device="cpu",
            model_configs=model_configs,
            tokenizer_config=ModelConfig(str(tokenizer_path)),
        )
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.pipe.freeze_except([])

        replace_linear_with_single_lora(
            self.pipe.dit,
            patterns=self.LORA_PATTERNS,
            rank=lora_rank,
            alpha=lora_rank,
            use_fp8=use_fp8,
        )
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]
        expected_lora_keys = {
            name
            for name, _ in self.named_parameters()
            if ".lora_A." in name or ".lora_B." in name
        }
        missing_lora_keys = sorted(expected_lora_keys.difference(state_dict))
        if missing_lora_keys:
            raise RuntimeError(
                "Checkpoint is missing LoRA weights; verify the model version and "
                f"--trained-ckpt (first keys: {missing_lora_keys[:10]})"
            )
        incompatible = self.load_state_dict(state_dict, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        if unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:10]}")

        self.start_timestep = start_timestep
        self.merge_lora_weights()
        self.requires_grad_(False)
        self.device = torch.device("cpu")

    def merge_lora_weights(self) -> None:
        """Replace every LoRA wrapper by its merged standard linear layer."""
        def merge(parent: torch.nn.Module) -> None:
            for name, module in list(parent.named_children()):
                if isinstance(module, SingleLoRALinear):
                    setattr(parent, name, module.merge())
                else:
                    merge(module)

        merge(self.pipe.dit)

    def move_to(self, device: str | torch.device) -> "Generator":
        self.device = torch.device(device)
        self.to(self.device)
        self.pipe.device = self.device
        return self

    @torch.no_grad()
    def infer(
        self,
        condition_image: Image.Image,
        *,
        prompt: str = "demoire the image",
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
        fidelity: float = 1.0,
        seed: int = 42,
        tiled: bool = True,
        tile_size: int = 128,
        tile_stride: int = 96,
    ) -> Image.Image:
        if not 0.0 <= fidelity <= 1.0:
            raise ValueError("fidelity must be in [0, 1]")

        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "input_image": None,
            "condition_image": condition_image,
            "height": condition_image.height,
            "width": condition_image.width,
            "seed": seed,
            "rand_device": self.device,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
            )

        condition_latents = inputs_shared["condition_latents"]
        noise = inputs_shared["noise"]
        positive = {
            "prompt_emb": inputs_posi["prompt_emb"],
            "prompt_emb_mask": inputs_posi["prompt_emb_mask"],
        }
        negative = {
            "prompt_emb": inputs_nega["prompt_emb"],
            "prompt_emb_mask": inputs_nega["prompt_emb_mask"],
        }

        timestep_id = torch.tensor([self.start_timestep], dtype=torch.long)
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(self.device)
        sigma = self.pipe.scheduler.sigmas[timestep_id].to(dtype=torch.bfloat16, device=self.device)

        fidelity_id = round(self.start_timestep + fidelity * (1000 - self.start_timestep))
        if fidelity_id < 1000:
            fidelity_index = torch.tensor([fidelity_id], dtype=torch.long)
            fidelity_timestep = self.pipe.scheduler.timesteps[fidelity_index].to(self.device)
            condition_latents = self.pipe.scheduler.add_noise(
                condition_latents.detach(), noise, fidelity_timestep
            )

        noisy_latents = self.pipe.scheduler.add_noise(
            inputs_shared["condition_latents"].detach(), noise, timestep
        )
        _, _, height, width = noisy_latents.shape
        model_args = {
            "height": height * 8,
            "width": width * 8,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }
        positive_prediction = self.pipe.model_fn(
            self.pipe.dit,
            noisy_latents,
            condition_latents,
            timestep,
            **positive,
            **model_args,
        )
        if cfg_scale == 1.0:
            prediction = positive_prediction
        else:
            negative_prediction = self.pipe.model_fn(
                self.pipe.dit,
                noisy_latents,
                condition_latents,
                timestep,
                **negative,
                **model_args,
            )
            prediction = negative_prediction + cfg_scale * (positive_prediction - negative_prediction)

        predicted_latents = noisy_latents - sigma * prediction
        image = self.pipe.vae.decode(
            predicted_latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return self.pipe.vae_output_to_image(image)
