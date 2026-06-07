#!/usr/bin/env python3
"""
Muestra frames originales vs reconstruidos para un checkpoint del VAE.

Elige un paciente por patologia (NOR/DCM/HCM/RV). 
Muestra 4 filas: original ED / recon ED / original ES / recon ES,
slice central (z=5) de cada volumen.

Uso:
    python src/plot_recon.py \
        --checkpoint       run_004/checkpoints/latest.pt \
        --preprocessed-dir /scratch/abernardo/preprocessed \
        --labels-csv       /home/abernardo/anita/info_data/ED_ES_frames_with_pathology.csv \
        --output           figures/recon_run004_latest.png
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import VAE3D, VAE3DDeep


PATHOLOGIES = ["NOR", "DCM", "HCM", "RV"]
PATHOLOGY_COLORS = {
    "NOR":  "#2196F3",
    "DCM":  "#F44336",
    "HCM":  "#4CAF50",
    "RV":   "#9C27B0",
}
ROW_LABELS = ["Original ED", "Recon ED", "Original ES", "Recon ES"]


def load_patient_info(csv_path: Path) -> dict[str, dict]:
    """
    Retorna {patient_stem: {"pathology": str, "ed": int, "es": int}}
    patient_stem = "{dataset}_{division}_{patient_id}"
    Indices ED/ES son 0-based (el CSV es 1-based, se resta 1).
    """
    info: dict[str, dict] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            stem = f"{row['dataset'].strip()}_{row['division'].strip()}_{row['patient_id'].strip()}"
            label = row["pathology"].strip().upper()
            if label == "ARV":
                label = "RV"
            info[stem] = {
                "pathology": label,
                "ed": int(row["ED_frame"]) - 1,   # 1-based -> 0-based
                "es": int(row["ES_frame"]) - 1,
            }
    return info


def pick_patients(pre_dir: Path, patient_info: dict[str, dict],
                  seed: int = 0) -> dict[str, dict]:
    
    rng = random.Random(seed)
    by_path: dict[str, list[dict]] = {p: [] for p in PATHOLOGIES}

    for stem, info in patient_info.items():
        label = info["pathology"]
        if label not in by_path:
            continue
        # Verificar que existen los frames ED y ES
        ed_pt = pre_dir / f"{stem}_t{info['ed']:03d}.pt"
        es_pt = pre_dir / f"{stem}_t{info['es']:03d}.pt"
        if ed_pt.exists() and es_pt.exists():
            by_path[label].append({"stem": stem, **info})

    chosen: dict[str, dict] = {}
    for label, candidates in by_path.items():
        if not candidates:
            continue
        train = [c for c in candidates if "_training_" in c["stem"]]
        pool  = train if train else candidates
        chosen[label] = rng.choice(pool)
    return chosen



def load_model(checkpoint: Path, arch: str, device: torch.device):
    ckpt  = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))

    if arch == "deep":
        latent_ch = state["encoder.conv_mu.weight"].shape[0]
        model = VAE3DDeep(latent_ch=latent_ch)
        print(f"VAE3DDeep  latent_ch={latent_ch}")
    else:
        latent_ch = state["encoder.conv_mu.weight"].shape[0]
        model = VAE3D(latent_ch=latent_ch)
        print(f"VAE3D  latent_ch={latent_ch}")

    model.load_state_dict(state)
    return model.eval().to(device)


@torch.no_grad()
def reconstruct(model, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Pasa el tensor por el VAE y retorna la reconstruccion."""
    x = tensor.unsqueeze(0).to(device)
    out = model(x)
    x_hat = out[0]   # primer elemento siempre es x_hat en todas las arquitecturas
    return x_hat.squeeze(0).cpu()


def load_frame(pre_dir: Path, stem: str, frame_idx: int) -> torch.Tensor:
    """Carga un frame preprocesado desde disco."""
    pt = pre_dir / f"{stem}_t{frame_idx:03d}.pt"
    return torch.load(pt, weights_only=True).float()  # (1,224,224,10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",       type=Path, required=True)
    ap.add_argument("--preprocessed-dir", type=Path, required=True)
    ap.add_argument("--labels-csv",       type=Path, required=True)
    ap.add_argument("--model-arch", type=str, default="v1",
                    choices=["v1", "deep"])
    ap.add_argument("--slice-z",  type=int, default=5,
                    help="Indice Z del slice a mostrar (0-9)")
    ap.add_argument("--output",   type=Path, default=Path("recon.png"))
    ap.add_argument("--seed",     type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, args.model_arch, device)
    print(f"Cargado: {args.checkpoint}")

    # Cargar CSV y elegir un paciente por patologia
    patient_info = load_patient_info(args.labels_csv)
    chosen = pick_patients(args.preprocessed_dir, patient_info, seed=args.seed)
    if not chosen:
        print("ERROR: no se encontraron pacientes con frames ED y ES disponibles.")
        return
    selected_str = ", ".join(f"{k}={v['stem']}" for k, v in chosen.items())
    print(f"Seleccionados: {selected_str}")

    # Figura: 4 filas x N cols
    cols_order = [p for p in PATHOLOGIES if p in chosen]
    n_cols = len(cols_order)
    fig, axes = plt.subplots(4, n_cols, figsize=(3.2 * n_cols, 13))
    if n_cols == 1:
        axes = axes[:, None]

    run_name = args.checkpoint.parent.parent.name
    ckpt_name = args.checkpoint.stem
    fig.suptitle(
        f"casero3 {run_name} ({ckpt_name}) - Input vs Reconstruction  (z={args.slice_z})",
        fontsize=12,
    )

    for col, label in enumerate(cols_order):
        pat   = chosen[label]
        color = PATHOLOGY_COLORS[label]
        stem  = pat["stem"]

        ed_tensor = load_frame(args.preprocessed_dir, stem, pat["ed"])
        es_tensor = load_frame(args.preprocessed_dir, stem, pat["es"])
        ed_recon  = reconstruct(model, ed_tensor, device)
        es_recon  = reconstruct(model, es_tensor, device)

        slz = args.slice_z
        ed_orig_sl  = ed_tensor[0, :, :, slz].numpy()
        ed_recon_sl = ed_recon[0,  :, :, slz].numpy()
        es_orig_sl  = es_tensor[0, :, :, slz].numpy()
        es_recon_sl = es_recon[0,  :, :, slz].numpy()

        # Mismo rango de intensidad para las 4 filas (referencia = original ED)
        vmin, vmax = ed_orig_sl.min(), ed_orig_sl.max()

        patient_id = stem.split("_")[-1]   # ej. "patient102"
        for row, img in enumerate([ed_orig_sl, ed_recon_sl, es_orig_sl, es_recon_sl]):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
                spine.set_visible(True)
            if row == 0:
                ax.set_title(f"{label}\n{patient_id}", color=color, fontsize=9)
            if col == 0:
                ax.set_ylabel(ROW_LABELS[row], fontsize=10)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=72, bbox_inches="tight")
    print(f"Guardado: {args.output}")


if __name__ == "__main__":
    main()
