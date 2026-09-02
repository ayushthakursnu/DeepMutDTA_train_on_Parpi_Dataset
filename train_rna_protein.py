"""
train_rna_protein.py

NEW FILE -- run this instead of main.py. It reuses the real (now RNA-patched)
DeepMutDTA building blocks -- model.TransformerModel, model_finetune.
SuperviseSimSiam, loss_func.RnCLoss / clf_contrastive_loss, utils.
get_data_loader -- but adds what main.py doesn't have and what you asked
for: per-epoch checkpoint/resume, a 5-seed sweep, accuracy/specificity/
recall on top of AUC/AUPR, an aggregate CSV sorted by val accuracy, and a
final single-sample inference report from the best checkpoint overall.

Place this file in the DeepMutDTA repo root (alongside the now-patched
model.py / utils.py, and the untouched main.py / fast_transformer.py /
model_finetune.py / loss_func.py / rna_encoder.py), after running
apply_patches.py and prepare_parpi_data.py.

Usage:
    python train_rna_protein.py --cell H9 \
        --train_path demo_data/clf_train_H9.csv \
        --test_path  demo_data/clf_test_H9.csv \
        --val_path   demo_data/clf_indepent_H9.csv
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from scipy.stats import pearsonr, spearmanr

from model import TransformerModel
from model_finetune import SuperviseSimSiam
from loss_func import clf_contrastive_loss
from utils import get_data_loader


# ============================================================================
# Args -- mirrors main.py's argparse defaults exactly (TransformerModel /
# FastformerEncoder read many of these directly), plus the new ones needed
# for the RNA branch, checkpointing, and the multi-seed sweep.
# ============================================================================
def build_args():
    p = argparse.ArgumentParser()
    # --- unchanged from the real main.py ---
    p.add_argument("--hidden_act", default="gelu")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_attention_heads", type=int, default=8)
    p.add_argument("--learn_rate", type=float, default=5e-5)
    p.add_argument("--layer_norm_eps", type=float, default=1e-12)
    p.add_argument("--hidden_dropout_prob", type=float, default=0.2)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_hidden_layers", type=int, default=2)
    p.add_argument("--vocab_size", type=int, default=31)  # protein branch
    p.add_argument("--max_seq_len", type=int, default=1000)  # protein branch
    p.add_argument("--max_drug_len", type=int, default=100)  # RNA branch length
    p.add_argument("--intermediate_size", type=int, default=256)
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--out_dim", type=int, default=1024)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.4)
    p.add_argument("--temp", type=float, default=3.0)
    p.add_argument("--task_type", type=str, default="clf")

    # --- new: RNA branch ---
    p.add_argument("--rna_vocab_size", type=int, default=6,
                    help="PAD, UNK, A, U, G, C")

    # --- new: data / run management ---
    p.add_argument("--cell", type=str, default="H9")
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--test_path", type=str, required=True)
    p.add_argument("--val_path", type=str, required=True)
    p.add_argument("--work_dir", type=str, default=None,
                    help="defaults to ./runs_<cell> if not set, so "
                         "different cell lines never collide")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--amp", action="store_true", default=True,
                    help="mixed precision -- roughly halves activation "
                         "memory for FastFormer's einsum-heavy attention")
    p.add_argument("--grad_accum_steps", type=int, default=1,
                    help="accumulate gradients over N steps to reach a "
                         "larger effective batch size without the memory "
                         "cost of a literally larger batch")
    p.add_argument("--log_every", type=int, default=50,
                    help="print a progress line every N training steps, "
                         "so a long epoch doesn't look like a hang")

    args = p.parse_args()
    if args.work_dir is None:
        args.work_dir = f"./runs_{args.cell}"
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# Metrics -- adds accuracy / specificity / recall on top of what utils.py's
# eval_clf() gives you (AUC/AUPR only), plus a point-biserial PCC/SCC of the
# predicted probability against the binary label. That last part is NOT the
# same thing as the paper's Kd/deltaG correlation (there is no continuous
# affinity anywhere in this data) -- it's reported because you asked for
# PCC/SCC, but read it as "does predicted probability rank-correlate with
# the binary label", not as a binding-affinity correlation.
# ============================================================================
def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    try:
        aupr = average_precision_score(y_true, y_prob)
    except ValueError:
        aupr = float("nan")

    if len(np.unique(y_true)) > 1 and len(np.unique(y_prob)) > 1:
        pcc, _ = pearsonr(y_true, y_prob)
        scc, _ = spearmanr(y_true, y_prob)
    else:
        pcc, scc = float("nan"), float("nan")

    return {
        "accuracy": accuracy, "specificity": specificity, "recall": recall,
        "auc": auc, "aupr": aupr, "pcc": pcc, "scc": scc,
    }


def run_eval(model, loader, amp=True):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for (smiles_padded, smiles_mask, seqs_wt_padded, seqs_wt_mask,
             label_wt, seqs_mt_padded, seqs_mt_mask, label_mt) in loader:
            smiles_padded = smiles_padded.cuda()
            smiles_mask = smiles_mask.cuda()
            seqs_wt_padded = seqs_wt_padded.cuda()
            seqs_wt_mask = seqs_wt_mask.cuda()
            seqs_mt_padded = seqs_mt_padded.cuda()
            seqs_mt_mask = seqs_mt_mask.cuda()

            with torch.autocast(device_type="cuda", enabled=amp):
                _, _, _, _, score_wt, score_mt = model(
                    smiles_padded, smiles_mask, seqs_wt_padded, seqs_wt_mask,
                    seqs_mt_padded, seqs_mt_mask,
                )
            y_true.extend(label_wt.numpy().tolist())
            y_true.extend(label_mt.numpy().tolist())
            y_prob.extend(score_wt.detach().float().cpu().numpy().tolist())
            y_prob.extend(score_mt.detach().float().cpu().numpy().tolist())
    return compute_metrics(y_true, y_prob)


# ============================================================================
# Checkpointing
# ============================================================================
def save_checkpoint(path, model, optimizer, scaler, epoch, best_val_acc, epochs_no_improve):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_val_acc": best_val_acc,
        "epochs_no_improve": epochs_no_improve,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }, path)


def load_checkpoint(path, model, optimizer, scaler):
    # weights_only=False: safe here because this is a checkpoint we wrote
    # ourselves (not a third-party download) -- it contains numpy/python
    # RNG state alongside the tensors, which PyTorch >=2.6's default
    # weights_only=True unpickler rejects.
    ckpt = torch.load(path, map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    try:
        # RNG state (esp. torch.cuda's) is not always portable across a
        # PyTorch/CUDA version or GPU change -- degrade gracefully rather
        # than aborting resume entirely, since the model/optimizer weights
        # (the part that actually matters) are restored above regardless.
        torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        np.random.set_state(ckpt["numpy_rng_state"])
        random.setstate(ckpt["python_rng_state"])
    except Exception as e:
        print(f"[warn] could not restore RNG state from checkpoint "
              f"(commonly happens after a Python/PyTorch/GPU change): {e}\n"
              f"[warn] continuing with a fresh random state -- model and "
              f"optimizer weights were restored fine; only exact "
              f"reproducibility of the random stream is affected.")
    return ckpt["epoch"], ckpt["best_val_acc"], ckpt["epochs_no_improve"]


# ============================================================================
# Train one seed
# ============================================================================
def train_one_seed(seed, args, train_loader, test_loader, val_loader):
    seed_dir = Path(args.work_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    results_path = seed_dir / "results.json"

    if args.resume and results_path.exists():
        prev = json.loads(results_path.read_text())
        prev_paths = prev.get("data_paths")
        current_paths = [args.train_path, args.test_path, args.val_path]
        if prev_paths is not None and prev_paths != current_paths:
            sys.exit(
                f"[FATAL] {results_path} exists but was trained on different "
                f"data than what you're pointing at right now.\n"
                f"  saved results used : {prev_paths}\n"
                f"  you just requested : {current_paths}\n"
                f"This looks like a stale results/checkpoint directory from "
                f"a different --cell or dataset. Either pass a different "
                f"--work_dir, or if you're sure the old run is no longer "
                f"needed: rm -rf {Path(args.work_dir)}"
            )
        print(f"[seed {seed}] already completed on matching data -> "
              f"loading saved results.")
        return prev

    set_seed(seed)

    backbone_model = TransformerModel(args=args).cuda()
    model = SuperviseSimSiam(backbone_model=backbone_model, args=args)
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.learn_rate}])
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    bce_loss = nn.BCELoss()

    last_ckpt = seed_dir / "last_checkpoint.pt"
    best_ckpt = seed_dir / "best_model.pt"

    start_epoch = 0
    best_val_acc = -1.0
    epochs_no_improve = 0
    if args.resume and last_ckpt.exists():
        start_epoch, best_val_acc, epochs_no_improve = load_checkpoint(
            last_ckpt, model, optimizer, scaler)
        start_epoch += 1
        print(f"[seed {seed}] resumed from epoch {start_epoch} "
              f"(best_val_acc so far = {best_val_acc:.4f})")

    best_val_metrics, best_test_metrics, best_epoch = None, None, -1

    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad()
        try:
            n_batches = len(train_loader)
        except TypeError:
            n_batches = None
        epoch_start = time.time()
        running_loss = 0.0
        running_count = 0
        for step, (smiles_padded, smiles_mask, seqs_wt_padded, seqs_wt_mask,
                   label_wt, seqs_mt_padded, seqs_mt_mask, label_mt) in enumerate(train_loader):
            smiles_padded = smiles_padded.cuda()
            smiles_mask = smiles_mask.cuda()
            seqs_wt_padded = seqs_wt_padded.cuda()
            seqs_wt_mask = seqs_wt_mask.cuda()
            label_wt = label_wt.cuda()
            seqs_mt_padded = seqs_mt_padded.cuda()
            seqs_mt_mask = seqs_mt_mask.cuda()
            label_mt = label_mt.cuda()

            with torch.autocast(device_type="cuda", enabled=args.amp):
                emb_view_1_wt, emb_view_2_wt, emb_view_1_mt, emb_view_2_mt, \
                    score_wt, score_mt = model(
                        smiles_padded, smiles_mask, seqs_wt_padded, seqs_wt_mask,
                        seqs_mt_padded, seqs_mt_mask,
                    )

            # Loss is computed OUTSIDE autocast, in fp32: nn.BCELoss (unlike
            # BCEWithLogitsLoss) is explicitly disallowed under autocast by
            # PyTorch, and since TransformerModel already applies sigmoid
            # internally, switching to BCEWithLogitsLoss isn't an option
            # here (it would double-apply sigmoid). Explicit .float() casts
            # ensure full fp32 precision for the loss math even though the
            # tensors came out of an autocast region.
            score_wt = score_wt.float()
            score_mt = score_mt.float()
            emb_view_1_wt = emb_view_1_wt.float()
            emb_view_2_wt = emb_view_2_wt.float()
            emb_view_1_mt = emb_view_1_mt.float()
            emb_view_2_mt = emb_view_2_mt.float()

            # identical loss composition to the real main.py's clf branch
            embeddings_1 = torch.cat([emb_view_1_wt, emb_view_2_mt], dim=0)
            labels_1 = torch.cat([label_wt, label_mt], dim=0)
            scl_loss_1 = clf_contrastive_loss(temp=args.temp, embedding=embeddings_1, label=labels_1)
            bce_loss_1 = bce_loss(score_wt, label_wt)

            embeddings_2 = torch.cat([emb_view_2_wt, emb_view_1_mt], dim=0)
            labels_2 = torch.cat([label_wt, label_mt], dim=0)
            bce_loss_2 = bce_loss(score_mt, label_mt)
            scl_loss_2 = clf_contrastive_loss(temp=args.temp, embedding=embeddings_2, label=labels_2)

            loss = args.alpha * (scl_loss_1 + scl_loss_2) + args.beta * (bce_loss_1 + bce_loss_2)
            loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * args.grad_accum_steps
            running_count += 1
            if (step + 1) % args.log_every == 0:
                elapsed = time.time() - epoch_start
                rate = running_count / elapsed  # batches/sec
                avg_loss = running_loss / running_count
                if n_batches:
                    eta_min = (n_batches - (step + 1)) / rate / 60 if rate > 0 else float("nan")
                    print(f"[seed {seed}] epoch {epoch:03d} | "
                          f"step {step + 1}/{n_batches} | "
                          f"avg_loss={avg_loss:.4f} | "
                          f"{rate:.2f} batch/s | "
                          f"ETA this epoch: {eta_min:.1f} min", flush=True)
                else:
                    print(f"[seed {seed}] epoch {epoch:03d} | "
                          f"step {step + 1} | avg_loss={avg_loss:.4f} | "
                          f"{rate:.2f} batch/s", flush=True)

        val_metrics = run_eval(model, val_loader, amp=args.amp)  # val_path = model-selection set
        print(f"[seed {seed}] epoch {epoch:03d} | "
              f"val_acc={val_metrics['accuracy']:.4f} val_auc={val_metrics['auc']:.4f} "
              f"val_aupr={val_metrics['aupr']:.4f}")

        improved = val_metrics["accuracy"] > best_val_acc
        if improved:
            best_val_acc = val_metrics["accuracy"]
            epochs_no_improve = 0
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_test_metrics = run_eval(model, test_loader, amp=args.amp)  # true held-out set
            torch.save(model.backbone.state_dict(), best_ckpt)
        else:
            epochs_no_improve += 1

        save_checkpoint(last_ckpt, model, optimizer, scaler, epoch, best_val_acc, epochs_no_improve)

        if epochs_no_improve >= args.patience:
            print(f"[seed {seed}] early stopping at epoch {epoch}")
            break

    result = {
        "seed": seed, "best_epoch": best_epoch,
        "data_paths": [args.train_path, args.test_path, args.val_path],
        "val": best_val_metrics, "test": best_test_metrics,
    }
    results_path.write_text(json.dumps(result, indent=2))
    return result


# ============================================================================
# Aggregate across seeds
# ============================================================================
def aggregate_results(all_results, work_dir):
    rows = []
    for r in all_results:
        row = {"seed": r["seed"], "best_epoch": r["best_epoch"]}
        for split in ("val", "test"):
            for metric, value in r[split].items():
                row[f"{split}_{metric}"] = value
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False).reset_index(drop=True)
    metric_cols = [c for c in df.columns if c.startswith("val_") or c.startswith("test_")]
    mean_row = {"seed": "mean", "best_epoch": ""}
    std_row = {"seed": "std", "best_epoch": ""}
    for c in metric_cols:
        mean_row[c] = df[c].mean()
        std_row[c] = df[c].std()
    df_out = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    out_path = Path(work_dir) / "results_aggregate.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n[aggregate] saved -> {out_path}")
    print(df_out.to_string(index=False))
    return df_out, out_path


# ============================================================================
# Final inference demo on 1 sample from the held-out test set
# ============================================================================
def run_inference_demo(best_seed, args, test_loader):
    backbone_model = TransformerModel(args=args).cuda()
    ckpt_path = Path(args.work_dir) / f"seed_{best_seed}" / "best_model.pt"
    backbone_model.load_state_dict(
        torch.load(ckpt_path, map_location="cuda", weights_only=False))
    backbone_model.eval()

    batch = next(iter(test_loader))
    (smiles_padded, smiles_mask, seqs_wt_padded, seqs_wt_mask,
     label_wt, seqs_mt_padded, seqs_mt_mask, label_mt) = batch

    smiles_padded, smiles_mask = smiles_padded[:1].cuda(), smiles_mask[:1].cuda()
    seqs_wt_padded, seqs_wt_mask = seqs_wt_padded[:1].cuda(), seqs_wt_mask[:1].cuda()

    with torch.no_grad():
        _, prob, _, _ = backbone_model(smiles_padded, seqs_wt_padded, smiles_mask, seqs_wt_mask)
    prob = prob.item()
    true_label = int(label_wt[0].item())
    pred_label = int(prob >= 0.5)

    print("\n" + "=" * 70)
    print("FINAL INFERENCE DEMO -- 1 sample from held-out TEST set")
    print("=" * 70)
    print(f"Best model : seed {best_seed} (checkpoint: {ckpt_path})")
    print(f"{'':28s}{'True':>15s}{'Predicted':>15s}")
    print(f"{'Bound (1) / unbound (0)':28s}{true_label:15d}{pred_label:15d}")
    print(f"{'P(bound)':28s}{'':15s}{prob:15.4f}")
    print(f"{'Correct?':28s}{'':15s}{'YES' if true_label == pred_label else 'NO':>15s}")
    print("=" * 70)


# ============================================================================
# Main
# ============================================================================
def main():
    args = build_args()
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    train_loader = get_data_loader(args.train_path, batch_size=args.batch_size, shuffle=True)
    test_loader = get_data_loader(args.test_path, batch_size=args.batch_size, shuffle=False)
    val_loader = get_data_loader(args.val_path, batch_size=args.batch_size, shuffle=False)

    all_results = []
    for seed in args.seeds:
        print(f"\n{'#'*70}\n# SEED {seed}\n{'#'*70}")
        all_results.append(train_one_seed(seed, args, train_loader, test_loader, val_loader))

    df_out, _ = aggregate_results(all_results, args.work_dir)
    seed_rows = df_out[(df_out["seed"] != "mean") & (df_out["seed"] != "std")]
    best_row = seed_rows.sort_values("val_accuracy", ascending=False).iloc[0]
    best_seed = int(best_row["seed"])
    print(f"\n[main] best seed overall by val_accuracy = {best_seed} "
          f"(val_accuracy={best_row['val_accuracy']:.4f})")

    run_inference_demo(best_seed, args, test_loader)


if __name__ == "__main__":
    main()