import nibabel as nib
import numpy as np
import torch
from monai.transforms import CenterSpatialCrop, SpatialPad
from scipy.ndimage import zoom

TARGET_H = 224
TARGET_W = 224
TARGET_D = 10
INPLANE_SPACING_MM = 1.5
TARGET_DZ_MM       = 10.0   # spacing Z objetivo (mm); cubre ~100mm del corazon


def _resample(frame: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """Resamplea X, Y y Z al spacing objetivo usando zooms del header."""
    scale_x = dx / INPLANE_SPACING_MM
    scale_y = dy / INPLANE_SPACING_MM
    scale_z = dz / TARGET_DZ_MM
    return zoom(frame, (scale_x, scale_y, scale_z), order=1)


def _crop_pad_to_target(frame: np.ndarray) -> np.ndarray:
    D = frame.shape[2]
    t = torch.from_numpy(frame[np.newaxis])

    # Center crop o pad D a TARGET_D
    if D > TARGET_D:
        start = (D - TARGET_D) // 2
        t = t[:, :, :, start : start + TARGET_D]
    elif D < TARGET_D:
        pad_b = (TARGET_D - D) // 2
        pad_a = TARGET_D - D - pad_b
        t = torch.nn.functional.pad(t, (pad_b, pad_a))

    # Crop si H/W muy largo, sino pad
    t = CenterSpatialCrop(roi_size=(TARGET_H, TARGET_W, TARGET_D))(t)
    t = SpatialPad(spatial_size=(TARGET_H, TARGET_W, TARGET_D))(t)
    return t.numpy()[0]


def _clip_to_01(frame: np.ndarray) -> np.ndarray:
    """Clip p1-p99 -> [0, 1]. Se aplica ANTES del crop/pad para que el padding
    con ceros coincida con el fondo real (aprox. 0)."""
    p1, p99 = np.percentile(frame, [1, 99])
    clipped = np.clip(frame, p1, p99)
    return (clipped - p1) / max(p99 - p1, 1e-8)


def _zscore(frame: np.ndarray) -> np.ndarray:
    mu, sigma = frame.mean(), frame.std()
    if sigma < 1e-8:
        sigma = 1e-8
    return (frame - mu) / sigma


def preprocess_volume(path: str) -> list[torch.Tensor]:
    """
    Preprocesa los cardiac MRI .nii.gz

    Por frame:
      1. Resample X/Y/Z a spacing fijo (1.5mm, 1.5mm, 10mm)
      2. Clip p1-p99 -> [0, 1]  (antes del padding; fondo queda en aprox. 0)
      3. Center crop/pad a (224, 224, 10)  (padding con ceros = fondo)
      4. Z-score

    Returns:
        List de T tensors, de shape (1, 224, 224, 10).
    """
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)

    if data.ndim == 3:
        data = data[..., np.newaxis]

    _, _, _, T = data.shape
    dx = float(img.header.get_zooms()[0])
    dy = float(img.header.get_zooms()[1])
    dz = float(img.header.get_zooms()[2])

    frames = []
    for t_idx in range(T):
        frame = data[:, :, :, t_idx]
        frame = _resample(frame, dx, dy, dz)
        frame = _clip_to_01(frame)
        frame = _crop_pad_to_target(frame)
        frame = _zscore(frame)
        frames.append(torch.from_numpy(frame[np.newaxis].copy()).float())

    return frames
