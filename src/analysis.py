"""
Analisis del espacio latente del VAE 3D entrenado.

Carga Z.npy + metadata.csv producidos por encode_all.py y genera:
  - Scatter PCA 2D (coloreado por patologia / dataset)
  - Scatter UMAP 2D
  - Trayectorias z(t) por paciente en el embedding 2D
  - Silhouette scores (por frame y por paciente)

Las etiquetas se cargan desde un CSV externo (--labels-csv) con columnas:
  dataset, division, patient_id, ED_frame, ES_frame, pathology

La clave de join es: {dataset}_{division.lower()}_{patient_id}
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Colores fijos para los 5 grupos cardiacos estandar; el resto usa tab20
_FIXED_PALETTE = {
    "NOR":  "#2196F3",
    "DCM":  "#F44336",
    "HCM":  "#4CAF50",
    "RV":   "#9C27B0",
}

# Aliases de patologia - se aplican antes de cualquier busqueda
_ALIASES = {"ARV": "RV"}


# -- etiquetas 

def load_labels_csv(csv_path: str | Path) -> dict[str, str]:
    """
    Construye {patient_stem: patologia} desde el CSV de etiquetas.

    patient_stem = "{dataset}_{division.lower()}_{patient_id}"
    ej: "ACDC_testing_patient114"

    Las filas con patologia NaN se ignoran.
    """
    df = pd.read_csv(csv_path)
    mapping = {}
    for _, row in df.iterrows():
        if pd.isna(row["pathology"]):
            continue
        label = _ALIASES.get(str(row["pathology"]), str(row["pathology"]))
        stem = f"{row['dataset']}_{str(row['division']).lower()}_{row['patient_id']}"
        mapping[stem] = label
    return mapping


def _patient_stem(patient_filename: str) -> str:
    """'ACDC_testing_patient114.nii.gz' -> 'ACDC_testing_patient114'"""
    return patient_filename.replace(".nii.gz", "")


def assign_labels(meta: pd.DataFrame, labels_dict: dict | None = None) -> pd.Series:
    """
    Etiqueta de patologia por frame.

    Prioridad:
      1. Busqueda en labels_dict por stem del paciente (desde CSV)
      2. Fallback: prefijo del dataset (ej. 'ACDC', 'DSB2')
    """
    def _label(patient: str) -> str:
        if labels_dict:
            stem = _patient_stem(patient)
            if stem in labels_dict:
                return labels_dict[stem]
        # fallback: prefijo del dataset desde el nombre de archivo
        return patient.split("_")[0]

    return meta["patient"].map(_label)


# -- reduccion de dimensionalidad 

def run_pca(Z: np.ndarray, n: int = 2) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n, random_state=5)
    return pca.fit_transform(Z), pca


def run_umap(Z: np.ndarray, n: int = 2) -> np.ndarray | None:
    import umap as umap_lib
    reducer = umap_lib.UMAP(n_components=n, random_state=5, n_neighbors=15, min_dist=0.1)
    return reducer.fit_transform(Z)


# -- visualizacion 

def _build_palette(unique_labels: list[str]) -> dict[str, tuple]:
    fixed = [l for l in unique_labels if l in _FIXED_PALETTE]
    other = [l for l in unique_labels if l not in _FIXED_PALETTE]
    colors = {l: _FIXED_PALETTE[l] for l in fixed}
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(other), 1)))
    for l, c in zip(other, cmap):
        colors[l] = c
    return colors


def plot_scatter(coords: np.ndarray, labels: pd.Series | list,
                 title: str, path: Path):
    labels = list(labels)
    unique = sorted(set(labels))
    colors = _build_palette(unique)

    fig, ax = plt.subplots(figsize=(11, 8))
    for lbl in unique:
        mask = np.array(labels) == lbl
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colors[lbl]], label=lbl, alpha=0.35, s=4, linewidths=0)
    ax.legend(markerscale=4, loc="best", fontsize=8,
              ncol=max(1, len(unique) // 10))
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Guardado: {path}")


def plot_trajectories(coords: np.ndarray, meta: pd.DataFrame,
                      title: str, path: Path):
    """
    Dibuja z(t) como curva por paciente etiquetado. El frame 0 se marca con un punto.
    Solo se dibujan los pacientes con etiqueta de patologia real.
    El resto forma el fondo gris.
    """
    if "label" not in meta.columns:
        print("No hay columna 'label' en meta - saltando trayectorias")
        return

    # Pacientes cuya etiqueta NO es solo el prefijo del dataset
    dataset_prefixes = set(meta["patient"].str.split("_").str[0])
    labeled_patients = [
        p for p in meta["patient"].unique()
        if meta.loc[meta["patient"] == p, "label"].iloc[0] not in dataset_prefixes
    ]

    if not labeled_patients:
        print("No se encontraron pacientes con etiqueta de patologia - saltando trayectorias")
        return

    unique_labels = sorted({
        meta.loc[meta["patient"] == p, "label"].iloc[0]
        for p in labeled_patients
    })
    colors = _build_palette(unique_labels)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(coords[:, 0], coords[:, 1],
               c="lightgrey", s=2, alpha=0.15, zorder=1)

    seen: set[str] = set()
    for patient in labeled_patients:
        lbl = meta.loc[meta["patient"] == patient, "label"].iloc[0]
        rows = meta[meta["patient"] == patient].sort_values("frame_idx")
        idx = rows.index.tolist()
        traj = coords[idx]
        c = colors[lbl]
        legend_lbl = lbl if lbl not in seen else "_nolegend_"
        seen.add(lbl)
        ax.plot(traj[:, 0], traj[:, 1], color=c, alpha=0.5,
                linewidth=0.7, label=legend_lbl, zorder=2)
        ax.scatter(traj[0, 0], traj[0, 1], color=c, s=12, zorder=3,
                   edgecolors="white", linewidths=0.4)

    ax.legend(loc="best", fontsize=8, ncol=max(1, len(unique_labels) // 10))
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Guardado: {path}")


# -- silhouette 

def compute_silhouette(Z: np.ndarray, meta: pd.DataFrame, out_dir: Path) -> dict | None:
    if "label" not in meta.columns:
        print("Sin etiquetas - saltando silhouette")
        return None

    dataset_prefixes = set(meta["patient"].str.split("_").str[0])
    labeled_mask = ~meta["label"].isin(dataset_prefixes)

    if labeled_mask.sum() < 10:
        print("No hay suficientes frames etiquetados para calcular silhouette")
        return None

    Z_lab = Z[labeled_mask]
    labels_frame = meta.loc[labeled_mask, "label"].values

    n_comp = min(50, Z_lab.shape[1], Z_lab.shape[0] - 1)
    Z_pca = PCA(n_components=n_comp, random_state=42).fit_transform(Z_lab)

    sil_frame = silhouette_score(Z_pca, labels_frame)
    print(f"\nSilhouette (por frame, PCA-{n_comp}): {sil_frame:.4f}")

    # Media por paciente
    lab_meta = meta[labeled_mask].copy()
    lab_meta["_row"] = np.where(labeled_mask)[0]
    patient_groups = lab_meta.groupby("patient")["_row"].apply(list)
    labels_patient = lab_meta.groupby("patient")["label"].first()

    Z_patient = np.stack([Z[rows].mean(0) for rows in patient_groups])
    lp = labels_patient.values
    n_unique = len(set(lp))

    if n_unique >= 2 and len(Z_patient) > n_unique:
        sil_patient = silhouette_score(Z_patient, lp)
        print(f"Silhouette (por paciente, PCA-{n_comp}): {sil_patient:.4f}")
    else:
        sil_patient = None
        print("Silhouette (por paciente): se necesitan mas pacientes por grupo")

    print("(rango -1 a 1; mayor = mejor separacion entre patologias)")

    results = {"silhouette_per_frame": sil_frame, "silhouette_per_patient": sil_patient}
    pd.DataFrame([results]).to_csv(out_dir / "silhouette.csv", index=False)
    print(f"Guardado: {out_dir / 'silhouette.csv'}")
    return results


# -- main ----------------------------------------------------------------------

def analyse(args):
    latent_dir = Path(args.latent_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Z = np.load(latent_dir / "Z.npy")
    meta = pd.read_csv(latent_dir / "metadata.csv")
    print(f"Z cargado: {Z.shape}  |  pacientes: {meta['patient'].nunique()}")

    # -- filtro por dataset --
    if args.datasets:
        wanted = set(args.datasets)
        keep = meta["dataset"].isin(wanted)
        if keep.sum() == 0:
            raise SystemExit(f"Ningun frame coincide con --datasets {args.datasets}. "
                             f"Disponibles: {sorted(meta['dataset'].unique())}")
        meta = meta[keep].reset_index(drop=True)
        Z = Z[keep.values]
        print(f"Filtro de dataset {args.datasets}: {len(meta)} frames conservados")

    labels_dict = None
    if args.labels_csv:
        labels_dict = load_labels_csv(args.labels_csv)
        n_matched = sum(
            1 for p in meta["patient"].unique()
            if _patient_stem(p) in labels_dict
        )
        print(f"CSV de etiquetas: {len(labels_dict)} entradas, "
              f"{n_matched}/{meta['patient'].nunique()} pacientes encontrados")

    labels = assign_labels(meta, labels_dict)
    meta["label"] = labels.values

    # -- filtro por clase --
    if args.classes:
        wanted = set(args.classes)
        keep = meta["label"].isin(wanted)
        if keep.sum() == 0:
            raise SystemExit(f"Ningun frame coincide con --classes {args.classes}. "
                             f"Disponibles: {sorted(meta['label'].unique())}")
        meta = meta[keep].reset_index(drop=True)
        Z = Z[keep.values]
        labels = meta["label"]
        print(f"Filtro de clase {args.classes}: {len(meta)} frames conservados")

    # PCA
    print("\n-- PCA --")
    pca_coords, _ = run_pca(Z, n=2)
    plot_scatter(pca_coords, labels, "PCA - espacio latente (todos los frames)",
                 out_dir / "pca_scatter.png")
    plot_trajectories(pca_coords, meta, "PCA - trayectorias del ciclo cardiaco",
                      out_dir / "pca_trajectories.png")

    # UMAP
    print("\n-- UMAP --")
    umap_coords = run_umap(Z, n=2)
    if umap_coords is not None:
        plot_scatter(umap_coords, labels, "UMAP - espacio latente (todos los frames)",
                     out_dir / "umap_scatter.png")
        plot_trajectories(umap_coords, meta, "UMAP - trayectorias del ciclo cardiaco",
                          out_dir / "umap_trajectories.png")

    # Silhouette
    print("\n-- Silhouette scores --")
    compute_silhouette(Z, meta, out_dir)

    # Guardar coordenadas como CSV
    pca_df = pd.concat([meta.reset_index(drop=True),
                        pd.DataFrame(pca_coords, columns=["pc1", "pc2"])], axis=1)
    pca_df.to_csv(out_dir / "pca_coords.csv", index=False)
    print(f"\nGuardado: {out_dir / 'pca_coords.csv'}")

    if umap_coords is not None:
        umap_df = pd.concat([meta.reset_index(drop=True),
                             pd.DataFrame(umap_coords, columns=["u1", "u2"])], axis=1)
        umap_df.to_csv(out_dir / "umap_coords.csv", index=False)
        print(f"Guardado: {out_dir / 'umap_coords.csv'}")

    print("\nAnalisis completo.")


def parse_args():
    p = argparse.ArgumentParser(description="Analizar el espacio latente del VAE")
    p.add_argument("--latent-dir",  type=str, default="latents")
    p.add_argument("--output-dir",  type=str, default="latents/figures")
    p.add_argument("--labels-csv",  type=str, default=None,
                   help="Ruta al CSV con columnas: dataset,division,patient_id,pathology")
    p.add_argument("--classes",  nargs="*", default=None,
                   help="Conservar solo estas patologias, ej: --classes NOR DCM HCM RV")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Conservar solo frames de estos datasets, ej: --datasets ACDC MNM1")
    return p.parse_args()


if __name__ == "__main__":
    analyse(parse_args())
