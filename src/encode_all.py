"""
Codifica todos los frames cardiacos en el espacio latente del VAE.

Carga el mejor checkpoint, corre encode() en cada frame de cada paciente,
y guarda una matriz Z de forma (N_pacientes * T, 128) mas metadata.

Archivos de salida (en --output-dir):
    Z.npy          array float32 (N_frames, latent_dim)
    metadata.csv   columnas: patient, frame_idx, dataset, path
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from dataset import DATASET_ROOT, CardiacFrameDataset, list_patients, split_patients_stratified
from model import VAE3D


def encode_all(args):
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # -- modelo --
    model = VAE3D(latent_dim=args.latent_dim).to(device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Checkpoint cargado: {ckpt_path}  (epoca {ckpt.get('epoch', '?')})")

    # -- pacientes --
    if args.split == "all":
        patients = list_patients(args.data_dir)
    else:
        train_p, val_p, test_p = split_patients_stratified(
            csv_path=args.pathology_csv,
            root=args.data_dir,
            seed=args.seed,
        )
        patients = {"train": train_p, "val": val_p, "test": test_p}[args.split]

    print(f"Codificando {len(patients)} pacientes ({args.split}) en {device} ...")

    # -- dataset -- sin cache, orden secuencial --
    ds = CardiacFrameDataset(patients, cache=False, preprocessed_dir=args.preprocessed_dir)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        sampler=SequentialSampler(ds),
    )

    # -- codificar --
    Z_chunks = []
    with torch.no_grad():
        for batch in dl:
            x = batch.to(device)
            mu, _ = model.encoder(x)
            Z_chunks.append(mu.cpu().float())

    Z = torch.cat(Z_chunks, dim=0).numpy()   # (N_frames, latent_dim)
    assert Z.shape == (len(ds), args.latent_dim), f"Shape inesperado de Z: {Z.shape}"

    # -- metadata --
    rows = []
    for path, frame_idx in ds.index:
        dataset_name = path.name.split("_")[0]
        rows.append(f"{path.name},{frame_idx},{dataset_name},{path}")

    # -- guardar --
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    z_path = out_dir / "Z.npy"
    np.save(z_path, Z)
    print(f"Z guardado: {z_path}  shape={Z.shape}")

    meta_path = out_dir / "metadata.csv"
    with open(meta_path, "w") as f:
        f.write("patient,frame_idx,dataset,path\n")
        f.write("\n".join(rows))
    print(f"Metadata guardada: {meta_path}  ({len(rows)} filas)")


def parse_args():
    p = argparse.ArgumentParser(description="Codificar todos los frames cardiacos con el VAE entrenado")
    p.add_argument("--checkpoint",   type=str, default="checkpoints/best.pt")
    p.add_argument("--data-dir",     type=str, default=str(DATASET_ROOT))
    p.add_argument("--output-dir",   type=str, default="latents")
    p.add_argument("--split",        type=str, default="all",
                   choices=["all", "train", "val", "test"])
    p.add_argument("--latent-dim",   type=int, default=128)
    p.add_argument("--batch-size",   type=int, default=16)
    p.add_argument("--num-workers",      type=int, default=0)
    p.add_argument("--preprocessed-dir", type=str, default=None,
                   help="Directorio con tensores .pt pre-guardados por preprocess_dataset.py")
    p.add_argument("--device",           type=str, default="auto")
    p.add_argument("--pathology-csv", type=str, default=None,
                   help="CSV requerido si --split no es 'all'.")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    encode_all(parse_args())
