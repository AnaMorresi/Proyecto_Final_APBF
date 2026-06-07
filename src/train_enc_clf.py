"""
Baseline: entrena solo Encoder + Clasificador con CrossEntropy.
Sin decoder, sin loss VAE. Solo supervisado con datos etiquetados.

Sirve para comparar si el VAE aporta algo al entrenamiento conjunto.

Uso:
    python src/train_enc_clf.py \
        --split-csv        checkpoints_mse/split.csv \
        --pathology-csv    .../ED_ES_frames_with_pathology.csv \
        --preprocessed-dir .../preprocessed \
        --output-dir       checkpoints_enc_clf
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from model import VAE3D, VAE3DDeep
from train_classifier import build_classifier, load_split, load_pathology, CLASSES, CLASS2IDX
from train_joint import EDESDataset


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -- Encoder + Clasificador --
    if args.model_arch == "deep":
        vae = VAE3DDeep(latent_ch=args.latent_ch).to(device)
    else:
        vae = VAE3D(latent_ch=args.latent_ch).to(device)
    in_ch   = args.latent_ch * 2          # ED + ES concatenados
    spatial = (14, 14, 1)
    clf = build_classifier(args.classifier_arch, n_classes=len(CLASSES),
                           in_ch=in_ch, spatial=spatial).to(device)

    # Solo el encoder -- el decoder no se usa
    encoder = vae.encoder
    for p in vae.decoder.parameters():
        p.requires_grad = False

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_clf = sum(p.numel() for p in clf.parameters())
    print(f"\n=== Encoder ({n_enc:,} params) ===")
    print(encoder)
    print(f"\n=== Clasificador ({n_clf:,} params) ===")
    print(clf)
    print(f"\n=== Configuracion ===")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print()

    # -- Datos --
    split_map = load_split(Path(args.split_csv))
    pat_info  = load_pathology(Path(args.pathology_csv))
    common    = {s for s in split_map if s in pat_info}
    pre_dir   = Path(args.preprocessed_dir)

    train_stems = sorted(s for s in common if split_map[s] == "train")
    val_stems   = sorted(s for s in common if split_map[s] == "val")
    test_stems  = sorted(s for s in common if split_map[s] == "test")
    print(f"Train: {len(train_stems)} | Val: {len(val_stems)} | Test: {len(test_stems)}")

    ds_tr = EDESDataset(train_stems, pat_info, pre_dir)
    ds_va = EDESDataset(val_stems,   pat_info, pre_dir)
    ds_te = EDESDataset(test_stems,  pat_info, pre_dir)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # -- Pesos por clase --
    labels_tr = torch.tensor([s[2] for s in ds_tr.samples])
    counts  = torch.bincount(labels_tr, minlength=len(CLASSES)).float()
    weights = (1.0 / counts.clamp(min=1)).to(device)
    weights = weights / weights.sum() * len(CLASSES)
    print("Pesos: " + " ".join(f"{CLASSES[i]}={weights[i]:.3f}" for i in range(len(CLASSES))))

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(clf.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    # -- Loop --
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "metrics.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_val_loss = float("inf")
    patience_ctr  = 0
    tr_losses, va_losses, tr_accs, va_accs = [], [], [], []

    for epoch in range(args.epochs):
        encoder.train(); clf.train()
        loss_sum = correct = total = 0

        for x_ed, x_es, y in dl_tr:
            x_ed, x_es, y = x_ed.to(device), x_es.to(device), y.to(device)
            mu_ed, _ = encoder(x_ed)
            mu_es, _ = encoder(x_es)
            z_cat  = torch.cat([mu_ed, mu_es], dim=1)
            logits = clf(z_cat)
            loss   = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item()
            correct  += (logits.argmax(1) == y).sum().item()
            total    += len(y)

        tr_loss = loss_sum / len(dl_tr)
        tr_acc  = correct / total

        encoder.eval(); clf.eval()
        loss_sum = correct = total = 0
        with torch.no_grad():
            for x_ed, x_es, y in dl_va:
                x_ed, x_es, y = x_ed.to(device), x_es.to(device), y.to(device)
                mu_ed, _ = encoder(x_ed)
                mu_es, _ = encoder(x_es)
                z_cat  = torch.cat([mu_ed, mu_es], dim=1)
                logits = clf(z_cat)
                loss_sum += criterion(logits, y).item()
                correct  += (logits.argmax(1) == y).sum().item()
                total    += len(y)

        va_loss = loss_sum / len(dl_va)
        va_acc  = correct / total

        tr_losses.append(tr_loss); va_losses.append(va_loss)
        tr_accs.append(tr_acc);    va_accs.append(va_acc)

        print(f"[{epoch:03d}] train loss={tr_loss:.4f} acc={tr_acc:.3f} | val loss={va_loss:.4f} acc={va_acc:.3f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.6f}", f"{tr_acc:.4f}",
                                           f"{va_loss:.6f}", f"{va_acc:.4f}"])

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_ctr = 0
            torch.save({"encoder": encoder.state_dict(), "clf": clf.state_dict()},
                       out_dir / "best.pt")
            print(f"  ** nuevo mejor val loss: {best_val_loss:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\nParada anticipada en epoca {epoch}")
                break

    # -- Graficos de loss --
    ep = range(len(tr_losses))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ep, tr_losses, label="train", color="steelblue")
    ax1.plot(ep, va_losses, label="val",   color="coral")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(ep, tr_accs, label="train", color="steelblue")
    ax2.plot(ep, va_accs, label="val",   color="coral")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "losses.png", dpi=100)
    plt.close()

    # -- Evaluacion en test --
    best = torch.load(out_dir / "best.pt", weights_only=True)
    encoder.load_state_dict(best["encoder"])
    clf.load_state_dict(best["clf"])
    encoder.eval(); clf.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for x_ed, x_es, y in dl_te:
            x_ed, x_es = x_ed.to(device), x_es.to(device)
            mu_ed, _ = encoder(x_ed)
            mu_es, _ = encoder(x_es)
            z_cat = torch.cat([mu_ed, mu_es], dim=1)
            all_preds.extend(clf(z_cat).argmax(1).cpu().tolist())
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split-csv",        type=str, required=True)
    p.add_argument("--pathology-csv",    type=str, required=True)
    p.add_argument("--preprocessed-dir", type=str, required=True)
    p.add_argument("--output-dir",       type=str, default="checkpoints_enc_clf")
    p.add_argument("--latent-ch",        type=int, default=64)
    p.add_argument("--model-arch",       type=str, default="v1",
                   choices=["v1", "deep"])
    p.add_argument("--classifier-arch",  type=str, default="flat_mlp",
                   choices=["conv_mlp", "mlp", "flat_mlp"])
    p.add_argument("--epochs",           type=int, default=100)
    p.add_argument("--batch-size",       type=int, default=16)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--patience",         type=int, default=15)
    p.add_argument("--weight-decay",     type=float, default=1e-4)
    p.add_argument("--num-workers",      type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
