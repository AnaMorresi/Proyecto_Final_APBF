"""
Entrenamiento conjunto semi-supervisado del VAE + clasificador.

Etiquetados:    L_VAE(ED) + L_VAE(ES) + lambda * CrossEntropy
No etiquetados: L_VAE(frame)
Total:          L_etiquetados + mu * L_no_etiquetados

Uso:
    # Desde cero:
    python src/train_joint.py \
        --split-csv        checkpoints_mse/split.csv \
        --pathology-csv    .../ED_ES_frames_with_pathology.csv \
        --preprocessed-dir .../preprocessed \
        --output-dir       checkpoints_joint

    # Desde pesos preentrenados:
    python src/train_joint.py \
        --checkpoint       checkpoints_mse/best.pt \
        --split-csv        checkpoints_mse/split.csv \
        --pathology-csv    .../ED_ES_frames_with_pathology.csv \
        --preprocessed-dir .../preprocessed \
        --output-dir       checkpoints_joint
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from model import VAE3D, vae_loss
from dataset import CardiacFrameDataset
from train_classifier import CardiacClassifier, load_split, load_pathology, CLASSES, CLASS2IDX


# -- Dataset 

class EDESDataset(Dataset):
    def __init__(self, stems: list[str], pat_info: dict, pre_dir: Path):
        self.samples = []
        skipped = 0
        for stem in stems:
            info = pat_info[stem]
            ed_pt = pre_dir / f"{stem}_t{info['ed']:03d}.pt"
            es_pt = pre_dir / f"{stem}_t{info['es']:03d}.pt"
            if ed_pt.exists() and es_pt.exists():
                self.samples.append((ed_pt, es_pt, CLASS2IDX[info["pathology"]]))
            else:
                skipped += 1
        if skipped:
            print(f"  Saltados {skipped} pacientes (archivos .pt no encontrados)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ed_pt, es_pt, label = self.samples[idx]
        x_ed = torch.load(ed_pt, weights_only=True).float()
        x_es = torch.load(es_pt, weights_only=True).float()
        return x_ed, x_es, torch.tensor(label, dtype=torch.long)


# -- Dataset no etiquetados 

class UnlabeledDataset(Dataset):
    """Frames de pacientes sin etiqueta de clase -- carga un frame por paciente."""
    def __init__(self, stems: list[str], labeled_stems: set[str], pre_dir: Path):
        self.pts = []
        for stem in stems:
            if stem in labeled_stems:
                continue
            pt = pre_dir / f"{stem}_t000.pt"
            if pt.exists():
                self.pts.append(pt)

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, idx):
        return torch.load(self.pts[idx], weights_only=True).float()


# -- Entrenamiento 

def run_epoch(vae, clf, loader, device, beta, lam, criterion,
              unlabeled_loader=None, mu=0.0, optimizer=None):
    training = optimizer is not None
    vae.train(training)
    clf.train(training)

    total_loss = vae_loss_sum = ce_loss_sum = unlabeled_loss_sum = 0.0
    correct = total = 0

    unlabeled_iter = iter(unlabeled_loader) if (training and unlabeled_loader) else None

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x_ed, x_es, y in loader:
            x_ed, x_es, y = x_ed.to(device), x_es.to(device), y.to(device)

            x_ed_hat, mu_ed, lv_ed = vae(x_ed)
            x_es_hat, mu_es, lv_es = vae(x_es)

            l_vae_ed, _, _ = vae_loss(x_ed, x_ed_hat, mu_ed, lv_ed, beta)
            l_vae_es, _, _ = vae_loss(x_es, x_es_hat, mu_es, lv_es, beta)
            l_vae = (l_vae_ed + l_vae_es) / 2

            z_cat = torch.cat([mu_ed, mu_es], dim=1)
            logits = clf(z_cat)
            l_ce = criterion(logits, y)

            loss = l_vae + lam * l_ce

            # No etiquetados
            l_unlabeled = torch.tensor(0.0, device=device)
            if unlabeled_iter is not None and mu > 0:
                try:
                    x_u = next(unlabeled_iter).to(device)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    x_u = next(unlabeled_iter).to(device)
                x_u_hat, mu_u, lv_u = vae(x_u)
                l_unlabeled, _, _ = vae_loss(x_u, x_u_hat, mu_u, lv_u, beta)
                loss = loss + mu * l_unlabeled

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss          += loss.item()
            vae_loss_sum        += l_vae.item()
            ce_loss_sum         += l_ce.item()
            unlabeled_loss_sum  += l_unlabeled.item()
            correct += (logits.argmax(1) == y).sum().item()
            total   += len(y)

    n = max(len(loader), 1)
    return total_loss / n, vae_loss_sum / n, ce_loss_sum / n, unlabeled_loss_sum / n, correct / total


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -- Modelo --
    vae = VAE3D(latent_ch=args.latent_ch).to(device)
    clf = CardiacClassifier(n_classes=len(CLASSES)).to(device)

    if args.checkpoint:
        ckpt  = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        vae.load_state_dict(state)
        print(f"Cargados pesos VAE desde {args.checkpoint}")
    else:
        print("Entrenando desde cero")

    # -- Mostrar modelo + configuracion --
    n_vae = sum(p.numel() for p in vae.parameters())
    n_clf = sum(p.numel() for p in clf.parameters())
    print(f"\n=== VAE ({n_vae:,} params) ===")
    print(vae.encoder)
    print(vae.decoder)
    print(f"\n=== Clasificador ({n_clf:,} params) ===")
    print(clf)
    print(f"\n=== Configuracion ===")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print(f"  clases: {CLASSES}")
    print()

    # -- Datos --
    split_map = load_split(Path(args.split_csv))
    pat_info  = load_pathology(Path(args.pathology_csv))
    common    = {s for s in split_map if s in pat_info}
    pre_dir   = Path(args.preprocessed_dir)

    train_stems = sorted(s for s in common if split_map[s] == "train")
    val_stems   = sorted(s for s in common if split_map[s] == "val")
    test_stems  = sorted(s for s in common if split_map[s] == "test")
    print(f"Train etiquetados: {len(train_stems)} | Val: {len(val_stems)} | Test: {len(test_stems)}")

    ds_tr = EDESDataset(train_stems, pat_info, pre_dir)
    ds_va = EDESDataset(val_stems,   pat_info, pre_dir)
    ds_te = EDESDataset(test_stems,  pat_info, pre_dir)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # -- No etiquetados (solo train) --
    dl_unlabeled = None
    if args.mu > 0:
        all_train_stems = sorted(s for s in split_map if split_map[s] == "train")
        labeled_set = set(train_stems)
        ds_unlabeled = UnlabeledDataset(all_train_stems, labeled_set, pre_dir)
        print(f"Train no etiquetados: {len(ds_unlabeled)}")
        dl_unlabeled = DataLoader(ds_unlabeled, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)

    # -- Pesos por clase --
    labels_tr = torch.tensor([s[2] for s in ds_tr.samples])
    counts  = torch.bincount(labels_tr, minlength=len(CLASSES)).float()
    weights = (1.0 / counts.clamp(min=1)).to(device)
    weights = weights / weights.sum() * len(CLASSES)
    print("Pesos: " + " ".join(f"{CLASSES[i]}={weights[i]:.3f}" for i in range(len(CLASSES))))

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        list(vae.parameters()) + list(clf.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    # -- Loop --
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "metrics.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_vae", "train_ce", "train_unlabeled", "train_acc",
                                          "val_loss",   "val_vae",   "val_ce",   "val_acc"])

    best_val_acc  = 0.0
    best_val_loss = float("inf")
    patience_ctr  = 0
    history = {"tr_loss": [], "va_loss": [], "tr_acc": [], "va_acc": []}

    for epoch in range(args.epochs):
        beta = min(args.beta_max, args.beta_max * epoch / max(args.beta_warmup, 1))

        tr_loss, tr_vae, tr_ce, tr_unlab, tr_acc = run_epoch(
            vae, clf, dl_tr, device, beta, args.lam, criterion,
            dl_unlabeled, args.mu, optimizer)
        va_loss, va_vae, va_ce, _, va_acc = run_epoch(
            vae, clf, dl_va, device, beta, args.lam, criterion)

        history["tr_loss"].append(tr_loss); history["va_loss"].append(va_loss)
        history["tr_acc"].append(tr_acc);   history["va_acc"].append(va_acc)

        print(f"[{epoch:03d}] beta={beta:.4f} | "
              f"train loss={tr_loss:.4f} (vae={tr_vae:.4f} ce={tr_ce:.4f} unlab={tr_unlab:.4f}) acc={tr_acc:.3f} | "
              f"val loss={va_loss:.4f} (vae={va_vae:.4f} ce={va_ce:.4f}) acc={va_acc:.3f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch,
                f"{tr_loss:.6f}", f"{tr_vae:.6f}", f"{tr_ce:.6f}", f"{tr_unlab:.6f}", f"{tr_acc:.4f}",
                f"{va_loss:.6f}", f"{va_vae:.6f}", f"{va_ce:.6f}", f"{va_acc:.4f}"])

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"vae": vae.state_dict(), "clf": clf.state_dict()},
                       out_dir / "best.pt")
            print(f"  ** nuevo mejor val acc: {best_val_acc:.3f}")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\nParada anticipada en epoca {epoch}")
                break

    # -- Graficos de loss --
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(len(history["tr_loss"]))
    ax1.plot(ep, history["tr_loss"], label="train", color="steelblue")
    ax1.plot(ep, history["va_loss"], label="val",   color="coral")
    ax1.set_title("Total Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(ep, history["tr_acc"], label="train", color="steelblue")
    ax2.plot(ep, history["va_acc"], label="val",   color="coral")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "losses.png", dpi=100)
    plt.close()

    # -- Evaluacion en test --
    best_ckpt = torch.load(out_dir / "best.pt", weights_only=True)
    vae.load_state_dict(best_ckpt["vae"])
    clf.load_state_dict(best_ckpt["clf"])
    vae.eval(); clf.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for x_ed, x_es, y in dl_te:
            x_ed, x_es = x_ed.to(device), x_es.to(device)
            _, mu_ed, _ = vae(x_ed)
            _, mu_es, _ = vae(x_es)
            z_cat  = torch.cat([mu_ed, mu_es], dim=1)
            logits = clf(z_cat)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_true.extend(y.tolist())

    print("\n-- Resultados en test --")
    print(classification_report(all_true, all_preds, target_names=CLASSES))

    cm = confusion_matrix(all_true, all_preds)
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
    plt.close()

    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out_dir / "confusion_matrix.csv")
    report = classification_report(all_true, all_preds, target_names=CLASSES, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(out_dir / "classification_report.csv")
    print(f"Guardado: {out_dir}")


# -- CLI 

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",       type=str, default=None,
                   help="VAE preentrenado (opcional)")
    p.add_argument("--split-csv",        type=str, required=True)
    p.add_argument("--pathology-csv",    type=str, required=True)
    p.add_argument("--preprocessed-dir", type=str, required=True)
    p.add_argument("--output-dir",       type=str, default="checkpoints_joint")
    p.add_argument("--latent-ch",        type=int, default=16)
    p.add_argument("--epochs",           type=int, default=100)
    p.add_argument("--batch-size",       type=int, default=16)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--lam",              type=float, default=1.0,
                   help="Peso de la loss de clasificacion")
    p.add_argument("--mu",               type=float, default=0.0,
                   help="Peso de la loss VAE de pacientes no etiquetados (0=desactivado)")
    p.add_argument("--beta-max",         type=float, default=0.001)
    p.add_argument("--beta-warmup",      type=int,   default=10)
    p.add_argument("--patience",         type=int,   default=15)
    p.add_argument("--weight-decay",     type=float, default=1e-4)
    p.add_argument("--num-workers",      type=int,   default=4)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
