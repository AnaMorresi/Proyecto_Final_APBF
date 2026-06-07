import csv
from collections import defaultdict
from pathlib import Path
import random

import nibabel as nib
import torch
from torch.utils.data import Dataset

from preprocessing import preprocess_volume

DATASET_ROOT = Path(r"G:\.shortcut-targets-by-id\1H1Il9DPlsz5VEHOWjaNo47Tf-arZebGw\CMR METADATASET - 2025\DATASET")
_EXCLUDE = {"Synthetic", "SCD", "dummy"}


def list_patients(root: Path | str = DATASET_ROOT) -> list[Path]:
    """Devuelve sorted list of .nii.gz files, excluye Synthetic, SCD y dummy."""
    return sorted(
        p for p in Path(root).glob("*.nii.gz")
        if p.name.split("_")[0] not in _EXCLUDE
    )


def split_patients_stratified(
    csv_path: Path | str,
    root: Path | str = DATASET_ROOT,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 5,
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Split 60-20-20 por (dataset, pathology).
    Cada (dataset, pathology) se divide proporcionalmente.
    Retorna (train, val, test) como listas de Path.
    """
    root = Path(root)
    rng = random.Random(seed)
    test_frac = 1.0 - train_frac - val_frac

    # Leer CSV y agrupar por (dataset, pathologia)
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    with open(str(csv_path), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds  = row["dataset"]
            div = row["division"]
            pid = row["patient_id"]
            pat = row["pathology"].strip() or "UNKNOWN"
            fpath = root / f"{ds}_{div}_{pid}.nii.gz"
            if fpath.exists():
                groups[(ds, pat)].append(fpath)

    train: list[Path] = []
    val:   list[Path] = []
    test:  list[Path] = []

    for (ds, pat), patients in sorted(groups.items()):
        shuffled = patients.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = round(n * test_frac)
        n_val  = round(n * val_frac)
        # garantizar que la suma no supere n
        if n_test + n_val >= n:
            n_test = max(0, n - 1)
            n_val  = 0
        test  += shuffled[:n_test]
        val   += shuffled[n_test:n_test + n_val]
        train += shuffled[n_test + n_val:]

    # DSB2 no tiene etiqueta de patologia -> solo train
    dsb2 = [p for p in list_patients(root) if p.name.startswith("DSB2_")]
    train += dsb2

    return train, val, test


class CardiacFrameDataset(Dataset):
    def __init__(self, patients: list[Path], cache: bool = False,
                 preprocessed_dir: Path | str | None = None):
        self._do_cache = cache
        self._cache: dict[str, list[torch.Tensor]] = {}
        self._pre_dir = Path(preprocessed_dir) if preprocessed_dir else None

        # Lee solo el header para descubrir T (cantidad de frames) por paciente
        self.index: list[tuple[Path, int]] = []
        skipped = []
        for p in patients:
            try:
                shape = nib.load(str(p)).shape
                T = shape[3] if len(shape) == 4 else 1
                for t in range(T):
                    self.index.append((p, t))
            except Exception:
                skipped.append(p.name)
        if skipped:
            print(f"Aviso: se saltaron {len(skipped)} archivos ilegibles: {skipped}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path, t = self.index[idx]
        try:
            return self._load(path, t)
        except Exception:
            # archivo corrupto - usar el siguiente sample valido
            return self.__getitem__((idx + 1) % len(self))

    def _load(self, path: Path, t: int) -> torch.Tensor:
        if self._pre_dir is not None:
            stem = path.name.replace(".nii.gz", "")
            pt_path = self._pre_dir / f"{stem}_t{t:03d}.pt"
            if pt_path.exists():
                return torch.load(pt_path, weights_only=True).float()

        key = str(path)
        if key in self._cache:
            return self._cache[key][t]

        frames = preprocess_volume(key)

        if self._do_cache:
            self._cache[key] = frames

        return frames[t]
