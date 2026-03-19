from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import cm

from src.models.transforms import get_transforms


def parse_fold_from_name(name: str):
    m = re.search(r"fold(\d+)", name)
    return int(m.group(1)) if m else None


def build_model(device: torch.device):
    from torchvision.models import densenet121, DenseNet121_Weights
    try:
        model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    except Exception:
        model = densenet121(weights=None)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)
    return model.to(device)


def denormalize_imagenet(img_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=img_tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=img_tensor.dtype).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()


def overlay_heatmap_on_image(img_tensor: torch.Tensor, cam: np.ndarray, alpha: float = 0.45):
    img = denormalize_imagenet(img_tensor)
    heatmap = cm.jet(cam)[..., :3]
    overlay = (1 - alpha) * img + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return img, heatmap, overlay


def load_test_row(project_root: Path, csv_path: Path, image_view: str, image_id: int):
    df = pd.read_csv(csv_path)
    path_col = "crop_path" if image_view == "crop" else "full_path"
    row = df[(df["split"] == "test") & (df["id"] == image_id)]
    if row.empty:
        raise ValueError(f"id {image_id} not found in test split")
    row = row.iloc[0]
    img_path = project_root / str(row[path_col])
    return row, img_path


def make_input_tensor(img_path: Path):
    img = Image.open(img_path).convert("RGB")
    transform = get_transforms("test")
    x = transform(img)
    return x


def gradcam_for_model(model, x, device):
    model.eval()

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    hook_f = model.features.register_forward_hook(forward_hook)
    hook_b = model.features.register_full_backward_hook(backward_hook)

    model.zero_grad()
    logits = model(x.unsqueeze(0).to(device)).squeeze(1)
    prob = torch.sigmoid(logits)[0].item()

    score = logits[0]
    score.backward(retain_graph=True)

    A = activations[0]
    dA = gradients[0]

    weights = dA.mean(dim=(2, 3), keepdim=True)
    cam = (weights * A).sum(dim=1)[0]
    cam = torch.relu(cam).detach().cpu().numpy()

    hook_f.remove()
    hook_b.remove()

    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    H, W = x.shape[1], x.shape[2]
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H), resample=Image.BILINEAR)
    cam = np.array(cam_img).astype(np.float32) / 255.0

    return cam, prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, required=True)
    ap.add_argument("--phase", type=str, required=True, choices=["head", "finetune"])
    ap.add_argument("--image_view", type=str, default="crop", choices=["crop", "full"])
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--alpha", type=float, default=0.45)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "processed" / "splits.csv"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = Path(args.runs_dir)
    run_folders = sorted([p for p in runs_dir.iterdir() if p.is_dir()])

    usable_runs = []
    for run in run_folders:
        ckpt = run / f"best_{args.phase}.pt"
        if ckpt.exists():
            usable_runs.append((parse_fold_from_name(run.name), ckpt))

    if not usable_runs:
        raise RuntimeError("No checkpoints found for selected phase")

    for image_id in args.ids:
        row, img_path = load_test_row(project_root, csv_path, args.image_view, image_id)
        x = make_input_tensor(img_path)

        cams = []
        probs = []

        for fold, ckpt_path in usable_runs:
            model = build_model(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])

            cam, prob = gradcam_for_model(model, x, device)
            cams.append(cam)
            probs.append(prob)

        cam_mean = np.mean(np.stack(cams, axis=0), axis=0)
        prob_mean = float(np.mean(probs))
        pred = int(prob_mean >= 0.5)

        img_np, heat_np, overlay_np = overlay_heatmap_on_image(x, cam_mean, alpha=args.alpha)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np)
        axes[0].set_title("Input")
        axes[0].axis("off")

        axes[1].imshow(heat_np)
        axes[1].set_title("Mean Grad-CAM")
        axes[1].axis("off")

        axes[2].imshow(overlay_np)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        title = (
            f"id={int(row['id'])} | y={int(row['label'])} | pred={pred} | prob={prob_mean:.3f} | "
            f"L={int(row['l_scc_class'])} | R={int(row['r_scc_class'])}"
        )
        fig.suptitle(title)
        fig.tight_layout()

        save_path = out_dir / f"ensemble_gradcam_id{image_id}_{args.phase}.png"
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()