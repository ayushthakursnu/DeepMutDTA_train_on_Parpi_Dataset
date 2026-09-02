"""
prepare_parpi_data.py

Converts PaRPI's dataset/clip_data/*_<CELL>.tsv files (RNA sequence,
structure, protein id, binary bound/unbound label -- one file per RBP
within that cell line) into the 5-column CSV format DeepMutDTA's
get_data_loader() expects:

    smile, seq_wt, seq_mt, label_wt, label_mt

IMPORTANT -- read this before trusting the output:
  PaRPI's data has no wild-type/mutant concept at all (it's single-RBP
  binding classification, not a mutation study). To route it through
  DeepMutDTA's SimSiam-MuTF machinery without fabricating a fake mutation,
  this script sets seq_wt == seq_mt (same protein sequence) and
  label_wt == label_mt (same binary label) for every row. Concretely this
  means:
    - The base model (FastFormer x2 + Top-K attention + fusion, i.e. the
      part you actually asked to "apply") trains completely normally
      exactly as if it were the wild-type-only DeepMutDTA pretraining
      stage, on the true bound/unbound signal.
    - The SimSiam-MuTF WT-vs-MT contrastive terms (RnC / SCL comparing the
      "wild-type" and "mutant" views) become a near no-op, since the two
      views are identical inputs -- there is no mutation effect for it to
      learn. This is an honest limitation of this data, not a bug.

Also note: I have not been able to directly inspect one of your actual
clip_data .tsv files (no internet/7z in my sandbox), so the column
auto-detection below is heuristic. It prints exactly what it detected --
check that output on your first run before trusting the CSVs it writes.

Usage:
    python prepare_parpi_data.py \
        --dataset_root /path/to/PaRPI/dataset \
        --cell H9 \
        --out_dir ./demo_data \
        --train_frac 0.8 --val_frac 0.1
"""
import argparse
import glob
import os
import random

import pandas as pd


def read_protein_fasta(path):
    seq_lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">") or not line:
                continue
            seq_lines.append(line)
    return "".join(seq_lines)


def load_clip_tsv(path):
    """Reads one PaRPI clip_data TSV and returns (rna_seqs, labels, protein_id).
    Column names are auto-detected -- check the printed diagnostic."""
    df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}

    def find(*keys):
        for k in keys:
            for lc, orig in cols.items():
                if k in lc:
                    return orig
        return None

    seq_col = find("seq", "sequence")
    label_col = find("label", "class", "target")
    protein_col = find("prot", "rbp")

    if seq_col is None or label_col is None or protein_col is None:
        raise RuntimeError(
            f"Could not auto-detect the required columns in {path}.\n"
            f"Columns present: {list(df.columns)}\n"
            f"Detected -> sequence: {seq_col}, label: {label_col}, "
            f"protein: {protein_col}\n"
            f"Open this file, check the real header names, and hard-code "
            f"them in load_clip_tsv() (replace the find(...) calls with "
            f"the literal column name strings)."
        )

    protein_id = df[protein_col].iloc[0]
    return (
        df[seq_col].astype(str).tolist(),
        df[label_col].astype(float).tolist(),
        protein_id,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True,
                     help="path to the extracted PaRPI 'dataset' folder "
                          "(contains clip_data/, protein/, esm/, dgl/, ...)")
    ap.add_argument("--cell", required=True, help="cell line, e.g. H9, K562")
    ap.add_argument("--out_dir", default="./demo_data")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    clip_dir = os.path.join(args.dataset_root, "clip_data")
    protein_dir = os.path.join(args.dataset_root, "protein")
    tsv_files = sorted(glob.glob(os.path.join(clip_dir, f"*_{args.cell}.tsv")))
    if not tsv_files:
        raise RuntimeError(f"No files matching *_{args.cell}.tsv found in "
                            f"{clip_dir}. Check --dataset_root and --cell.")

    print(f"Found {len(tsv_files)} RBP dataset file(s) for cell line "
          f"'{args.cell}':")
    for f in tsv_files:
        print("  -", f)

    rows = []
    for tsv in tsv_files:
        rna_seqs, labels, protein_id = load_clip_tsv(tsv)
        fasta_path = os.path.join(protein_dir, f"{protein_id}.fasta")
        if not os.path.exists(fasta_path):
            print(f"[warn] {fasta_path} not found -- skipping {tsv}")
            continue
        protein_seq = read_protein_fasta(fasta_path)
        for rna_seq, label in zip(rna_seqs, labels):
            rows.append({
                "smile": rna_seq,        # RNA sequence. Kept as 'smile' so
                                          # DeepMutDTA's get_data_loader()
                                          # needs no further changes.
                "seq_wt": protein_seq,
                "seq_mt": protein_seq,    # no WT/MT distinction in this data
                "label_wt": float(label),
                "label_mt": float(label),
            })
        print(f"  {os.path.basename(tsv)}: {len(rna_seqs)} samples "
              f"(protein={protein_id}, protein_len={len(protein_seq)})")

    if not rows:
        raise RuntimeError("No usable rows were produced -- check the "
                            "[warn] messages above.")

    df = pd.DataFrame(rows)
    print(f"\nTotal samples for cell line '{args.cell}': {len(df)}")
    print(f"Label balance (label_wt): "
          f"{df['label_wt'].value_counts().to_dict()}")

    rng = random.Random(args.seed)
    idx = list(range(len(df)))
    rng.shuffle(idx)
    n_train = int(len(idx) * args.train_frac)
    n_val = int(len(idx) * args.val_frac)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, f"clf_train_{args.cell}.csv")
    test_path = os.path.join(args.out_dir, f"clf_test_{args.cell}.csv")
    val_path = os.path.join(args.out_dir, f"clf_indepent_{args.cell}.csv")
    df.iloc[train_idx].to_csv(train_path, index=False)
    df.iloc[test_idx].to_csv(test_path, index=False)
    df.iloc[val_idx].to_csv(val_path, index=False)

    print("\nWrote:")
    print(f"  {train_path}  ({len(train_idx)} rows)")
    print(f"  {test_path}  ({len(test_idx)} rows)  "
          f"<- used for model selection every epoch (DeepMutDTA's own "
          f"convention: this file plays the role of a validation set)")
    print(f"  {val_path}  ({len(val_idx)} rows)  "
          f"<- true held-out set, only evaluated when the test metric "
          f"improves; this is DeepMutDTA's actual final reported result")


if __name__ == "__main__":
    main()