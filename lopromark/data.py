from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageFolderDataset(Dataset):
    """Small image-folder dataset for localized watermark training."""

    def __init__(self, root: str | Path, image_size: int = 256) -> None:
        self.root = Path(root)
        self.paths = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found under {self.root}")
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image)


def random_rectangle_masks(
    batch_size: int,
    num_regions: int,
    height: int,
    width: int,
    *,
    min_frac: float = 0.18,
    max_frac: float = 0.45,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create simple non-empty rectangular region masks [B, R, H, W]."""
    masks = torch.zeros(batch_size, num_regions, height, width, device=device)
    min_h = max(4, int(height * min_frac))
    max_h = max(min_h + 1, int(height * max_frac))
    min_w = max(4, int(width * min_frac))
    max_w = max(min_w + 1, int(width * max_frac))

    for b in range(batch_size):
        for r in range(num_regions):
            rh = random.randint(min_h, min(max_h, height))
            rw = random.randint(min_w, min(max_w, width))
            top = random.randint(0, max(0, height - rh))
            left = random.randint(0, max(0, width - rw))
            masks[b, r, top : top + rh, left : left + rw] = 1.0
    return masks


def split_batch_regions(images: torch.Tensor, region_masks: torch.Tensor) -> torch.Tensor:
    """Return a binary union mask for convenience."""
    del images
    return (region_masks.sum(dim=1, keepdim=True) > 0).float()
