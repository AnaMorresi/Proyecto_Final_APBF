"""
Entrenamiento del clasificador sobre latentes congelados del encoder VAE.

Usa solo frames ED y ES por paciente.
Clases: NOR, DCM, HCM, RV

Uso:
    python src/train_classifier.py \
        --checkpoint       checkpoints_mse/best.pt \
        --split-csv        checkpoints_mse/split.csv \
        --pathology-csv    /home/abernardo/anita/info_data/ED_ES_frames_with_pathology.csv \
        --preprocessed-dir /home/abernardo/anita/preprocessed \
        --output-dir       checkpoints_classifier
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from model import VAE3D, VAE3DDeep

CLASSES   = ["NOR", "DCM", "HCM", "RV"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
ALIASES   = {"ARV": "RV"}
SKIP      = {"MINF"}


# -- Clasificadores 

class CardiacClassifierConvMLP(nn.Module):
    """Conv3d + MLP sobre el espacio latente."""
    def __init__(self, n_classes: int = 4, in_ch: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.mlp  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pool(self.conv(z)))


class CardiacClassifierMLP(nn.Module):
    """Solo pool global + MLP sobre el espacio latente."""
    def __init__(self, n_classes: int = 4, in_ch: int = 32):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.mlp  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pool(z))


class CardiacClassifierFlatMLP(nn.Module):
    """Flatten completo del latente + MLP. Sin pool -- preserva info espacial."""
    def __init__(self, n_classes: int = 4, in_ch: int = 32,
                 spatial: tuple = (14, 14, 1)):
        super().__init__()
        flat_dim = in_ch * spatial[0] * spatial[1] * spatial[2]
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


# alias para compatibilidad con train_joint.py
CardiacClassifier = CardiacClassifierConvMLP


def build_classifier(arch: str, n_classes: int, in_ch: int,
                     spatial: tuple = (14, 14, 1)) -> nn.Module:
    if arch == "mlp":
        return CardiacClassifierMLP(n_classes=n_classes, in_ch=in_ch)
    if arch == "flat_mlp":
        return CardiacClassifierFlatMLP(n_classes=n_classes, in_ch=in_ch,
                                        spatial=spatial)
    return CardiacClassifierConvMLP(n_classes=n_classes, in_ch=in_ch)


# -- Carga de datos 

def load_split(split_csv: Path) -> dict[str, str]:
    """Retorna {filename_stem: split}."""
    mapping = {}
    with open(split_csv, newline="") as f:
        for row in csv.DictReader(f):
            stem = row["filename"].replace(".nii.gz", "")
            mapping[stem] = row["split"]
    return mapping


def load_pathology(pathology_csv: Path) -> dict[str, dict]:
    """Retorna {stem: {pathology, ed, es}} con indices de frame 0-based."""
    info = {}
    with open(pathology_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pat = ALIASES.get(row["pathology"].strip(), row["pathology"].strip())
            if pat in SKIP or pat not in CLASS2IDX:
                continue
            stem    = f"{row['dataset'].strip()}_{row['division'].strip()}_{row['patient_id'].strip()}"
            dataset = row["dataset"].strip()
            # ACDC usa indices 1-based; MNM1/MNM2/SSC usan 0-based
            offset  = 1 if dataset == "ACDC" else 0
            info[stem] = {
                "pathology": pat,
                "ed": int(row["ED_frame"]) - offset,
                "es": int(row["ES_frame"]) - offset,
            }
    return info


@torch.no_grad()
def encode_patients(
    encoder: nn.Module,
    stems: list[str],
    pat_info: dict,
    pre_dir: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Retorna (Z, labels) donde Z shape (N, 32, 14, 14, 1) y labels (N,).
    Omite pacientes con archivos .pt faltantes.
    """
    Z_list, y_list = [], []
    skipped = 0
    for stem in stems:
        info = pat_info[stem]
        ed_pt = pre_dir / f"{stem}_t{info['ed']:03d}.pt"
        es_pt = pre_dir / f"{stem}_t{info['es']:03d}.pt"
        if not ed_pt.exists() or not es_pt.exists():
            skipped += 1
            continue
        x_ed = torch.load(ed_pt, weights_only=True).float().unsqueeze(0).to(device)
        x_es = torch.load(es_pt, weights_only=True).float().unsqueeze(0).to(device)
        z_ed, _ = encoder(x_ed)
        z_es, _ = encoder(x_es)
        z_cat = torch.cat([z_ed, z_es], dim=1).squeeze(0).cpu()  # (32, 14, 14, 1)
        Z_list.append(z_cat)
        y_list.append(CLASS2IDX[info["pathology"]])
    if skipped:
        print(f"  Saltados {skipped} pacientes (archivos .pt no encontrados)")
    return torch.stack(Z_list), torch.tensor(y_list, dtype=torch.long)


# -- Entrenamiento 

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -- Cargar encoder congelado --
    ckpt  = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    latent_ch = state["encoder.conv_mu.weight"].shape[0]
    if args.model_arch == "deep":
        vae = VAE3DDeep(latent_ch=latent_ch).to(device)
    else:
        vae = VAE3D(latent_ch=latent_ch).to(device)
    vae.load_state_dict(state)
    vae.encoder.eval()
    for p in vae.encoder.parameters():
        p.requires_grad = False
    print(f"Encoder cargado (arch={args.model_arch}, latent_ch={latent_ch}, congelado)")

    # -- Cargar metadata --
    split_map  = load_split(Path(args.split_csv))
    pat_info   = load_pathology(Path(args.pathology_csv))

    # Solo pacientes que aparecen en ambos CSV
    common = {s for s in split_map if s in pat_info}
    print(f"Pacientes con split + etiqueta de patologia: {len(common)}")

    train_stems = sorted(s for s in common if split_map[s] == "train")
    val_stems   = sorted(s for s in common if split_map[s] == "val")
    test_stems  = sorted(s for s in common if split_map[s] == "test")
    print(f"Train: {len(train_stems)} | Val: {len(val_stems)} | Test: {len(test_stems)}")

    # -- Pre-codificar --
    pre_dir = Path(args.preprocessed_dir)
    print("Codificando train...")
    Z_tr, y_tr = encode_patients(vae.encoder, train_stems, pat_info, pre_dir, device)
    print("Codificando val...")
    Z_va, y_va = encode_patients(vae.encoder, val_stems,   pat_info, pre_dir, device)
    print("Codificando test...")
    Z_te, y_te = encode_patients(vae.encoder, test_stems,  pat_info, pre_dir, device)
    print(f"Z_train: {Z_tr.shape} | Z_val: {Z_va.shape} | Z_test: {Z_te.shape}")

    # -- Dataloaders --
    dl_tr = DataLoader(TensorDataset(Z_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    dl_va = DataLoader(TensorDataset(Z_va, y_va), batch_size=args.batch_size)
    dl_te = DataLoader(TensorDataset(Z_te, y_te), batch_size=args.batch_size)

    # -- Clasificador --
    in_ch = Z_tr.shape[1]   # latent_ch * 2 (ED + ES concatenados)
    spatial = tuple(Z_tr.shape[2:])   # (14, 14, 1)
    clf = build_classifier(args.classifier_arch, n_classes=len(CLASSES),
                           in_ch=in_ch, spatial=spatial).to(device)
    n_params = sum(p.numel() for p in clf.parameters())
    print(f"Params del clasificador: {n_params:,}")

    # -- Pesos por clase (inverso de frecuencia) --
    counts = torch.bincount(y_tr, minlength=len(CLASSES)).float()
    weights = (1.0 / counts.clamp(min=1)).to(device)
    weights = weights / weights.sum() * len(CLASSES)
    print("Pesos por clase: " + " ".join(f"{CLASSES[i]}={weights[i]:.3f}" for i in range(len(CLASSES))))

    optimizer = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=weights)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0

    log_path = out_dir / "metrics.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    all_tr_loss, all_va_loss = [], []

    # -- Loop --
    for epoch in range(args.epochs):
        clf.train()
        correct = total = 0
        loss_sum = 0.0
        for Z_b, y_b in dl_tr:
            Z_b, y_b = Z_b.to(device), y_b.to(device)
            logits = clf(Z_b)
            loss = criterion(logits, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            correct  += (logits.argmax(1) == y_b).sum().item()
            total    += len(y_b)
        tr_loss = loss_sum / len(dl_tr)
        tr_acc  = correct / total

        clf.eval()
        correct = total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for Z_b, y_b in dl_va:
                Z_b, y_b = Z_b.to(device), y_b.to(device)
                logits = clf(Z_b)
                loss_sum += criterion(logits, y_b).item()
                correct  += (logits.argmax(1) == y_b).sum().item()
                total    += len(y_b)
        va_loss = loss_sum / len(dl_va)
        va_acc  = correct / total

        all_tr_loss.append(tr_loss)
        all_va_loss.append(va_loss)

        print(f"[{epoch:03d}] train loss: {tr_loss:.4f} acc: {tr_acc:.3f} | val loss: {va_loss:.4f} acc: {va_acc:.3f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.6f}", f"{tr_acc:.4f}", f"{va_loss:.6f}", f"{va_acc:.4f}"])

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(clf.state_dict(), out_dir / "best_classifier.pt")
            print(f"  ** nuevo mejor val acc: {best_val_acc:.3f}")

        # Parada anticipada
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nParada anticipada en epoca {epoch} (sin mejora en val loss por {args.patience} epocas)")
                break

    # -- Grafico de loss --
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs_range = range(len(all_tr_loss))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs_range, all_tr_loss, label="train loss", color="steelblue")
    ax.plot(epochs_range, all_va_loss, label="val loss",   color="coral")
    best_ep = int(torch.tensor(all_va_loss).argmin().item())
    ax.axvline(x=best_ep, color="gray", linestyle=":", linewidth=1.2,
               label=f"early stop ref (ep {best_ep})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("CrossEntropy Loss")
    ax.set_title("Classifier -- Train vs Val Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "losses.png", dpi=100)
    print(f"Guardado: {out_dir / 'losses.png'}")
    plt.close()

    # -- Evaluacion en test --
    clf.load_state_dict(torch.load(out_dir / "best_classifier.pt", weights_only=True))
    clf.eval()
    all_preds, all_true, all_probs = [], [], []
    with torch.no_grad():
        for Z_b, y_b in dl_te:
            Z_b = Z_b.to(device)
            logits = clf(Z_b)
            all_probs.extend(torch.softmax(logits, dim=1).cpu().tolist())
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_true.extend(y_b.tolist())

    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score
    import numpy as np

    y_bin  = label_binarize(all_true, classes=list(range(len(CLASSES))))
    y_prob = np.array(all_probs)
    auc_per_class = {}
    for i, cls in enumerate(CLASSES):
        auc_per_class[cls] = roc_auc_score(y_bin[:, i], y_prob[:, i])
    auc_macro = roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro')
    auc_weighted = roc_auc_score(y_bin, y_prob, multi_class='ovr', average='weighted')

    print("\n-- Resultados en test --")
    print(classification_report(all_true, all_preds, target_names=CLASSES))
    print("AUC-ROC por clase:")
    for cls, auc in auc_per_class.items():
        print(f"  {cls}: {auc:.4f}")
    print(f"AUC-ROC macro:    {auc_macro:.4f}")
    print(f"AUC-ROC weighted: {auc_weighted:.4f}")

    cm = confusion_matrix(all_true, all_preds)
    print("Matriz de confusion:")
    print(cm)

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # CSV
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

    report = classification_report(all_true, all_preds, target_names=CLASSES, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    for cls, auc in auc_per_class.items():
        report_df.loc[cls, 'auc_roc'] = auc
    report_df.loc['macro avg', 'auc_roc']    = auc_macro
    report_df.loc['weighted avg', 'auc_roc'] = auc_weighted
    report_df.to_csv(out_dir / "classification_report.csv")

    # Imagen
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title("Confusion Matrix - Test set")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=100)


# -- CLI 

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",       type=str, required=True)
    p.add_argument("--split-csv",        type=str, required=True)
    p.add_argument("--pathology-csv",    type=str, required=True)
    p.add_argument("--preprocessed-dir", type=str, required=True)
    p.add_argument("--output-dir",       type=str, default="checkpoints_classifier")
    p.add_argument("--epochs",           type=int, default=50)
    p.add_argument("--patience",         type=int, default=10,
                   help="Parada anticipada: epocas sin mejora en val loss")
    p.add_argument("--batch-size",       type=int, default=32)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--model-arch",       type=str, default="v1",
                   choices=["v1", "deep"])
    p.add_argument("--classifier-arch",  type=str, default="conv_mlp",
                   choices=["conv_mlp", "mlp", "flat_mlp"])
    p.add_argument("--weight-decay",     type=float, default=1e-4)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
