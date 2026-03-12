from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

from src.models.dataset_multitask import DatasetConfig, TIDSMastitisMultitaskDataset
from src.models.multitask_densenet import DenseNet121Multitask
from src.models.transforms import get_transforms



def parse_fold_from_name(name: str):
    m = re.search(r"fold(\d+)", name)
    return int(m.group(1)) if m else None


def find_run_for_fold(runs_dir: Path, fold: int) -> Path:
    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and f"fold{fold}" in p.name]
    if not candidates:
        raise FileNotFoundError(f"No run found for fold {fold} in {runs_dir}")
    # toma el más reciente
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def overlay_heatmap_on_image(img_tensor, cam, alpha=0.45):
    """
    img_tensor: [3,H,W] in [0,1] approx after transform without imagenet denorm
    cam: [H,W] normalized 0..1
    """
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)

    heatmap = cm.jet(cam)[..., :3]  # RGB
    overlay = (1 - alpha) * img + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return img, heatmap, overlay


def build_model(device):
    model = DenseNet121Multitask(pretrained=False).to(device)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--phase", type=str, default="finetune", choices=["head", "finetune"])
    ap.add_argument("--image_view", type=str, default="crop", choices=["crop", "full"])
    ap.add_argument("--target", type=str, default="binary", choices=["binary", "left", "right"])
    ap.add_argument("--num_images", type=int, default=6)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "processed" / "splits.csv"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Test dataset
    test_ds = TIDSMastitisMultitaskDataset(
        DatasetConfig(
            project_root=project_root,
            csv_path=csv_path,
            image_view=args.image_view,
            split="test",
            fold=None,
            mode="test",
        ),
        transform=get_transforms("test"),
        
    )

    run = find_run_for_fold(Path(args.runs_dir), args.fold)
    ckpt_path = run / f"best_{args.phase}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    model = build_model(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Hook sobre el último mapa convolucional
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    hook_f = model.features.register_forward_hook(forward_hook)
    hook_b = model.features.register_full_backward_hook(backward_hook)

    # Elegimos algunas imágenes del test
    n = min(args.num_images, len(test_ds))

    for idx in range(n):
        img, y = test_ds[idx]
        x = img.unsqueeze(0).to(device)
        model.zero_grad()

        activations.clear()
        gradients.clear()

        outputs = model(x)

        if args.target == "binary":
            score = outputs["logits_bin"][0]
            target_name = f"bin_y{int(y['y_bin'].item())}"
        elif args.target == "left":
            cls_idx = int(y["y_left"].item())
            score = outputs["logits_left"][0, cls_idx]
            target_name = f"left_cls{cls_idx+1}"
        else:  # right
            cls_idx = int(y["y_right"].item())
            score = outputs["logits_right"][0, cls_idx]
            target_name = f"right_cls{cls_idx+1}"

        score.backward(retain_graph=True)

        A = activations[0]         # [1,C,H,W]
        dA = gradients[0]          # [1,C,H,W]

        weights = dA.mean(dim=(2, 3), keepdim=True)   # [1,C,1,1]
        cam = (weights * A).sum(dim=1, keepdim=False) # [1,H,W]
        cam = torch.relu(cam)

        cam = cam[0].detach().cpu().numpy()

        # Normalizar CAM
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Redimensionar al tamaño de la imagen transformada
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (img.shape[2], img.shape[1]), resample=Image.BILINEAR
        )
        cam = np.array(cam_img).astype(np.float32) / 255.0

        orig, heat, overlay = overlay_heatmap_on_image(img, cam, alpha=0.45)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(orig)
        axes[0].set_title("Input")
        axes[0].axis("off")

        axes[1].imshow(heat)
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        title = (
            f"id={int(y['id'].item())} | "
            f"bin={int(y['y_bin'].item())} | "
            f"L={int(y['y_left'].item())+1} | "
            f"R={int(y['y_right'].item())+1}"
        )
        fig.suptitle(title)
        fig.tight_layout()

        save_path = out_dir / f"gradcam_{args.target}_{target_name}_idx{idx}_id{int(y['id'].item())}.png"
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    hook_f.remove()
    hook_b.remove()

    print(f"Saved Grad-CAM figures to: {out_dir}")


if __name__ == "__main__":
    main()