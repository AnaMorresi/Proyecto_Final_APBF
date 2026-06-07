"""
Pre-procesa todos los pacientes una vez y los guarda como tensors.

Nombres: {stem}_t{t:03d}.pt por frame.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from dataset import list_patients
from preprocessing import preprocess_volume


def _process_one(path_str: str, out_dir_str: str) -> tuple[str, int]:
    path = Path(path_str)
    out_dir = Path(out_dir_str)
    stem = path.name.replace(".nii.gz", "")

    frames = preprocess_volume(path_str)
    for t, frame in enumerate(frames):
        out_path = out_dir / f"{stem}_t{t:03d}.pt"
        if not out_path.exists():
            torch.save(frame.half(), out_path)   # float16

    return stem, len(frames)


def preprocess_all(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patients = list_patients(args.data_dir)
    print(f"Total pacientes: {len(patients)}")

    # Skip patients ya completamente preprocesados
    todo = [p for p in patients
            if not (out_dir / f"{p.name.replace('.nii.gz','')}_t000.pt").exists()]

    if not todo:
        print("Todos los pacientes ya preprocesados.")
        return

    print(f"Quedan: {len(todo)} pacientes -> guardando en {out_dir}\n")

    total_frames = 0
    done = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {
            ex.submit(_process_one, str(p), str(out_dir)): p
            for p in todo
        }
        for fut in as_completed(futures):
            stem, n = fut.result()
            total_frames += n
            done += 1
            if done % 100 == 0 or done == len(todo):
                print(f"{done}/{len(todo)} pacientes  |  {total_frames} frames guardados")

    print(f"\n{total_frames} frames en {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Preprocesar dataset a tensores .pt")
    p.add_argument("--data-dir",    type=str, required=True,
                   help="Directorio con los .nii.gz")
    p.add_argument("--out-dir",     type=str, required=True,
                   help="Directorio de salida para los tensores .pt")
    p.add_argument("--num-workers", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    preprocess_all(parse_args())
