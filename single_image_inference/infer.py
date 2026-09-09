"""Command-line entry point for exactly one input image and one output image."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from PIL import Image


def adaptive_pad(image: Image.Image, tile_size: int, stride: int) -> Image.Image:
    """Pad the right and bottom so tiled inference covers the complete image."""
    width, height = image.size
    pad_height = (
        tile_size - height
        if height <= tile_size
        else ((height - tile_size + stride - 1) // stride) * stride + tile_size - height
    )
    pad_width = (
        tile_size - width
        if width <= tile_size
        else ((width - tile_size + stride - 1) // stride) * stride + tile_size - width
    )
    if pad_width == 0 and pad_height == 0:
        return image
    padded = Image.new(image.mode, (width + pad_width, height + pad_height), color=0)
    padded.paste(image, (0, 0))
    return padded


@torch.no_grad()
def run(args: argparse.Namespace) -> Path:
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("--input and --output must be different paths")
    if args.scale <= 0:
        raise ValueError("--scale must be greater than zero")
    if not 0.0 <= args.fidelity <= 1.0:
        raise ValueError("--fidelity must be in [0, 1]")
    if not 0 <= args.start_timestep < 1000:
        raise ValueError("--start-timestep must be in [0, 1000)")
    if args.tile_size <= 0 or args.tile_stride <= 0 or args.tile_stride > args.tile_size:
        raise ValueError("tile sizes must satisfy: 0 < tile_stride <= tile_size")

    if __package__:
        from .generator import Generator
    else:  # Support: python path/to/infer.py
        from generator import Generator

    component_root = args.component_path or args.qwen_path
    model = Generator(
        transformer_root=args.qwen_path,
        component_root=component_root,
        checkpoint=args.trained_ckpt,
        torch_dtype=torch.bfloat16,
        start_timestep=args.start_timestep,
    ).move_to(args.device)

    with Image.open(input_path) as source:
        image = source.convert("RGB")
    output_size = (round(image.width * args.scale), round(image.height * args.scale))
    if min(output_size) < 1:
        raise ValueError("--scale makes the output width or height smaller than one pixel")
    resized = image.resize(output_size, Image.Resampling.BICUBIC)
    padded = adaptive_pad(
        resized,
        tile_size=args.tile_size * 8,
        stride=args.tile_stride * 8,
    )
    result = model.infer(
        padded,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        cfg_scale=args.cfg,
        fidelity=args.fidelity,
        seed=args.seed,
        tiled=True,
        tile_size=args.tile_size,
        tile_stride=args.tile_stride,
    )
    result = result.crop((0, 0, *output_size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-step Qwen-Image inference for one input image and one output image."
    )
    parser.add_argument("--input", required=True, help="Input image file.")
    parser.add_argument("--output", required=True, help="Output image file.")
    parser.add_argument("--trained-ckpt", required=True, help="Trained generator checkpoint (.pth).")
    parser.add_argument(
        "--qwen-path",
        default=os.environ.get("qwen_path"),
        required=not os.environ.get("qwen_path"),
        help="Model root providing transformer/ (default: qwen_path environment variable).",
    )
    parser.add_argument(
        "--component-path",
        help="Optional separate model root providing text_encoder/, vae/ and tokenizer/.",
    )
    parser.add_argument("--prompt", default="demoire the image")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--fidelity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--tile-stride", type=int, default=96)
    parser.add_argument("--start-timestep", type=int, default=750)
    return parser.parse_args()


if __name__ == "__main__":
    saved_path = run(parse_args())
    print(f"Saved output to: {saved_path}")
