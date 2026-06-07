import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import DATASET_ROOT, CardiacFrameDataset, split_patients_stratified
from model import VAE3D, VAE3DDeep, vae_loss


# -- programa de beta 

def get_beta(epoch: int, warmup: int, beta_max: float = 1.0) -> float:
    """Warmup lineal de 0 a beta_max en `warmup` epocas."""
    if warmup <= 0:
        return beta_max
    return min(beta_max, beta_max * epoch / warmup)


# -- helpers por epoca 

def _run_epoch(model, loader, device, beta, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = recon_sum = kl_sum = 0.0
    mu_list, lv_list = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x in loader:
            x = x.to(device)
            x_hat, mu, log_var = model(x)
            total, recon, kl = vae_loss(x, x_hat, mu, log_var, beta)

            if training:
                optimizer.zero_grad()
                total.backward()
                optimizer.step()
            else:
                mu_list.append(mu.detach().cpu())
                lv_list.append(log_var.detach().cpu())

            loss_sum  += total.item()
            recon_sum += recon.item()
            kl_sum    += kl.item()

    n = max(len(loader), 1)
    active_dims = None
    if not training and mu_list:
        mu_all = torch.cat(mu_list, dim=0)
        lv_all = torch.cat(lv_list, dim=0)
        kl_per_dim = -0.5 * (1 + lv_all - mu_all.pow(2) - lv_all.exp()).mean(
            dim=[i for i in range(lv_all.ndim) if i != 1]
        )
        active_dims = (kl_per_dim > 0.1).sum().item()

    return loss_sum / n, recon_sum / n, kl_sum / n, active_dims


# -- logging 

def _init_logger(args):
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(Path(args.checkpoint_dir) / "tb_logs"))
        print("Logger: tensorboard")
        return writer
    except Exception:
        pass

    log_path = Path(args.checkpoint_dir) / "metrics.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a")
    if log_path.stat().st_size == 0:
        fh.write("epoch,beta,train_loss,train_mse,train_kl,val_loss,val_mse,val_kl\n")
    print(f"Logger: CSV -> {log_path}")
    return fh


def _log(logger, metrics: dict):
    if hasattr(logger, "add_scalar"):
        ep = metrics["epoch"]
        for k, v in metrics.items():
            if k != "epoch":
                logger.add_scalar(k, v, ep)
    elif hasattr(logger, "write"):
        ep = metrics["epoch"]
        row = (
            f"{ep},{metrics['beta']:.4f},"
            f"{metrics['train/loss']:.6f},{metrics['train/mse']:.6f},{metrics['train/kl']:.6f},"
            f"{metrics['val/loss']:.6f},{metrics['val/mse']:.6f},{metrics['val/kl']:.6f}\n"
        )
        logger.write(row)
        logger.flush()


def _close_logger(logger):
    if hasattr(logger, "close"):
        logger.close()


# -- checkpointing 

def _save(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        torch.save(state, tmp)
        tmp.rename(path)
    except Exception as e:
        print(f"Aviso: fallo al guardar checkpoint {path.name}: {e}")
        tmp.unlink(missing_ok=True)


# -- split CSV 

def _save_split_csv(path: Path, train_p, val_p, test_p):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "split"])
        for p in train_p:
            w.writerow([p.name, "train"])
        for p in val_p:
            w.writerow([p.name, "val"])
        for p in test_p:
            w.writerow([p.name, "test"])
    print(f"Split guardado -> {path}  (train={len(train_p)}, val={len(val_p)}, test={len(test_p)})")


# -- funcion principal de entrenamiento 

def train(args):
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt_dir = Path(args.checkpoint_dir)

    # -- datos --
    train_p, val_p, test_p = split_patients_stratified(
        csv_path=args.pathology_csv,
        root=args.data_dir,
        train_frac=0.6,
        val_frac=0.2,
        seed=args.seed,
    )
    _save_split_csv(ckpt_dir / "split.csv", train_p, val_p, test_p)

    train_ds = CardiacFrameDataset(train_p, cache=args.cache, preprocessed_dir=args.preprocessed_dir)
    val_ds   = CardiacFrameDataset(val_p,   cache=args.cache, preprocessed_dir=args.preprocessed_dir)

    dl_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                 pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))
    train_dl = DataLoader(train_ds, shuffle=True,  **dl_kw)
    val_dl   = DataLoader(val_ds,   shuffle=False, **dl_kw)

    # -- modelo --
    if args.model_arch == "deep":
        model = VAE3DDeep(latent_ch=args.latent_ch).to(device)
    else:
        model = VAE3D(latent_ch=args.latent_ch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # -- reanudar --
    start_epoch   = 0
    best_val_loss = float("inf")
    latest_ckpt   = ckpt_dir / "latest.pt"
    if args.resume and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"Reanudando desde epoca {ckpt['epoch']} (mejor val={best_val_loss:.4f})")

    logger = _init_logger(args)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nDevice: {device} | Params: {n_params:,}")
    print(f"Frames train: {len(train_ds)} | Frames val: {len(val_ds)}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr} | beta warmup: {args.beta_warmup}\n")

    # -- loop --
    for epoch in range(start_epoch, args.epochs):
        beta = get_beta(epoch, args.beta_warmup, args.beta_max)
        t0   = time.time()

        tr_loss, tr_mse, tr_kl, _           = _run_epoch(model, train_dl, device, beta, optimizer)
        va_loss, va_mse, va_kl, active_dims = _run_epoch(model, val_dl,   device, beta)

        elapsed = time.time() - t0
        print(
            f"[{epoch:03d}] beta={beta:.4f} | "
            f"train {tr_loss:.4f} (mse={tr_mse:.4f} kl={tr_kl:.4f}) | "
            f"val {va_loss:.4f} (mse={va_mse:.4f} kl={va_kl:.4f}) | "
            f"active: {active_dims}/{args.latent_ch} | {elapsed:.0f}s"
        )

        metrics = {
            "epoch": epoch, "beta": beta,
            "train/loss": tr_loss, "train/mse": tr_mse, "train/kl": tr_kl,
            "val/loss":   va_loss, "val/mse":   va_mse, "val/kl":   va_kl,
        }
        _log(logger, metrics)

        state = dict(epoch=epoch, model=model.state_dict(),
                     optimizer=optimizer.state_dict(),
                     best_val_loss=best_val_loss, args=vars(args))
        _save(state, ckpt_dir / "latest.pt")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            state["best_val_loss"] = best_val_loss
            _save(state, ckpt_dir / "best.pt")
            print(f"  ** nuevo mejor val loss: {best_val_loss:.4f}")

    _close_logger(logger)
    print("\nEntrenamiento completo.")


# -- CLI 

def parse_args():
    p = argparse.ArgumentParser(description="Entrenar 3D VAE en frames de CMR cardiaca")
    p.add_argument("--data-dir",      type=str,   default=str(DATASET_ROOT))
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch-size",    type=int,   default=4)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--latent-ch",     type=int,   default=16)
    p.add_argument("--beta-warmup",   type=int,   default=0)
    p.add_argument("--beta-max",      type=float, default=0.001)
    p.add_argument("--weight-decay",  type=float, default=0.0)
    p.add_argument("--num-workers",   type=int,   default=0)
    p.add_argument("--device",        type=str,   default="auto")
    p.add_argument("--checkpoint-dir",type=str,   default="checkpoints")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--cache",             action="store_true")
    p.add_argument("--preprocessed-dir", type=str, default=None,
                   help="Directorio con tensores .pt pre-guardados por preprocess_dataset.py")
    p.add_argument("--pathology-csv",  type=str, required=True,
                   help="CSV con columnas dataset/division/patient_id/pathology.")
    p.add_argument("--model-arch",    type=str,   default="v1",
                   choices=["v1", "deep"],
                   help="v1=CNN 2 bloques stride-4 | deep=CNN 4 bloques stride-2")
    p.add_argument("--resume",        action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
