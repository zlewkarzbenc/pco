import argparse
import subprocess
import shutil
import os
import pandas as pd
import joblib
from Bio import SeqIO 
import time
import traceback

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

def run_nsp3(query_path, output_path, model_path):
    """Run the nsp3.py script on a batch file."""
    cmd = [
        "python", "NetSurfP-3.0_standalone/nsp3.py",
        "-i", query_path,
        "-o", output_path,
        "-m", model_path,
    ]
    subprocess.run(cmd, check=True)


def compute_features(input_path, output_path):
    """Run features.py on the CSV output of nsp3."""
    cmd = [
        "python", "features.py",
        "-i", input_path, # this should be the result of nsp3
        "-o", output_path,
    ]
    subprocess.run(cmd, check=True)


def make_predictions(features_csv, model_path, output_csv, fasta_file):
    """Make predictions using the XGBoost model."""
    # Load the model
    model = joblib.load(model_path)
    
    # Load features
    df = pd.read_csv(features_csv)
    X = df.iloc[:, 1:].values  # Assuming features start from column 1
    headers = df['id']
    
    # Extract sequences from the FASTA file
    sequences = {record.id: str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")}
    
    # Make predictions
    predictions = model.predict_proba(X)[:, 1]
    
    # Create results DataFrame
    results = pd.DataFrame({
        "id": headers,
        "crystallizablity score": predictions,
        "sequence": headers.map(sequences)  # Map sequences using the FASTA headers
    })
    
    # Save results
    results.to_csv(output_csv, index=False)

def clean_up(directory):
    """Delete the file tree created by NetSurfP-3.0."""
    if os.path.exists(directory):
        shutil.rmtree(directory)

# The following two functions exist only to sort the final result file. 
# Because of how NetSurfP works, the sequences in it's output are not in the same order as in the query.
# What you see below is my effort to bring back the original order.
def _normalize_id(s: str) -> str:
    """
    Normalize sequence IDs for robust matching.
    - strip leading '>' if present
    - take first whitespace-separated token if no delimiters
    - prefer tokens split by '|' or '$' when present
    """
    if s is None:
        return ""
    s = str(s).strip()
    if s.startswith(">"):
        s = s[1:]
    # if pipe or dollar delimiter appear, prefer the second token if it looks like an accession
    for delim in ("|", "$"):
        if delim in s:
            parts = s.split(delim)
            # prefer the part that looks like an accession (alphanumeric)
            for p in parts:
                if p and any(ch.isalnum() for ch in p):
                    token = p
                    return token.split()[0]
            # fallback to first non-empty
            token = parts[0]
            return token.split()[0]
    # no special delimiter: take first whitespace-separated token
    return s.split()[0]

def postprocess_predictions(pred_csv_path: str, fasta_path: str, out_csv_path: str):
    """
       Attempt to reorder predictions to match FASTA order and overwrite pred_csv_path.
       If any step fails, leave the original (unsorted) CSV and print an informative message.
    """
    # 0) Quick checks
    if not os.path.exists(pred_csv_path):
        raise FileNotFoundError(f"Predictions CSV not found: {pred_csv_path}")

    # 1) Read predictions CSV as-is and keep a copy (this ensures an unsorted file is always available)
    try:
        unsorted_df = pd.read_csv(pred_csv_path)
    except Exception as e:
        raise ValueError(f"Failed to read predictions CSV ({pred_csv_path}): {e}")

    # Immediately ensure the unsorted CSV has at least some columns and write it back (normalize)
    try:
        # If the predictions were written in other formats, this will canonicalize column names only when saving
        unsorted_df.to_csv(out_csv_path, index=False)
    except Exception as e:
        # If writing unsorted fails, abort early
        raise IOError(f"Failed to write unsorted predictions CSV to {out_csv_path}: {e}")

    # 2) Attempt the sorting procedure; any failure results in keeping the unsorted CSV
    try:
        # Read FASTA to preserve order and sequences
        ordered_ids = []
        seq_map = {}
        for rec in SeqIO.parse(fasta_path, "fasta"):
            norm = _normalize_id(rec.id)
            if not norm:
                norm = _normalize_id(rec.description)
            ordered_ids.append(norm)
            seq_map[norm] = str(rec.seq)
        if len(ordered_ids) == 0:
            raise ValueError("No sequences found in provided FASTA file.")

        # Re-read predictions (using the copy we just saved so formats are normalized)
        df = pd.read_csv(out_csv_path, dtype=str)

        # Detect id and score columns robustly
        cols_lowercase = [c.lower() for c in df.columns]
        # id column candidates
        id_col = None
        for cand in ("id", "seq_id", "sequence_id", "name", "identifier", "header"):
            if cand in cols_lowercase:
                id_col = df.columns[cols_lowercase.index(cand)]
                break
        if id_col is None:
            id_col = df.columns[0]  # fallback to first column

        # score column candidates
        score_col = None
        for cand in ("crystallizability_score", "score", "prob", "probability", "prediction", "pred", "y_pred"):
            if cand in cols_lowercase:
                score_col = df.columns[cols_lowercase.index(cand)]
                break
        if score_col is None:
            # attempt to find first numeric-like column that is not id_col
            for c in df.columns:
                if c == id_col:
                    continue
                sample = df[c].dropna().astype(str)
                if len(sample) == 0:
                    continue
                try:
                    float(sample.iloc[0].replace(",", "."))
                    score_col = c
                    break
                except Exception:
                    continue
        if score_col is None:
            # fallback: use second column if present
            if df.shape[1] >= 2:
                score_col = df.columns[1]
            else:
                raise ValueError("Could not detect a score column in predictions CSV.")

        # Build mapping norm_id -> score
        pred_map = {}
        for _, row in df.iterrows():
            raw_id = str(row[id_col])
            norm_id = _normalize_id(raw_id)
            if not norm_id:
                continue
            raw_val = row[score_col]
            # try to coerce to float, else keep as string
            try:
                val = float(str(raw_val).strip().replace(",", "."))
            except Exception:
                val = raw_val
            pred_map[norm_id] = val

        # Build ordered rows using FASTA order, attempt approximate matching if direct fails
        out_rows = []
        for nid in ordered_ids:
            score = pred_map.get(nid, None)
            seq = seq_map.get(nid, "")
            if score is None:
                # attempt alternate matches (suffix/prefix) to be tolerant about header differences
                for k in pred_map:
                    if k.endswith(nid) or nid.endswith(k) or nid.startswith(k) or k.startswith(nid):
                        score = pred_map.get(k)
                        break
            out_rows.append({"id": nid, "crystallizability_score": score, "sequence": seq})

        sorted_df = pd.DataFrame(out_rows, columns=["id", "crystallizability_score", "sequence"])

        # Overwrite original predictions CSV with the sorted version
        sorted_df.to_csv(out_csv_path, index=False)
    
    except Exception as e:
        print("Failed to sort predictions according to FASTA order. Sadly, your results are now shuffled due to NetSurf's inner workings :(")
        print("Sorting error details:")
        traceback.print_exc()
        # leave unsorted CSV as-is (out_csv_path already contains the unsorted copy)


def main():
    parser = argparse.ArgumentParser(description="Make protein crystallizability prediction on your fasta file")
    parser.add_argument("-i", "--input", required=True, help="Path to your query (FASTA format).")
    parser.add_argument("-o", "--output", help="Path to output directory.", default='./')
    args = parser.parse_args()

    # Define paths
    nsp3_model_path = 'NetSurfP-3.0_standalone/models/nsp3.pth'
    nsp3_output_dir = os.path.join(args.output, "nsp3_output")
    nsp_result_path = os.path.join(nsp3_output_dir, '01', "01.csv")
    features_csv_path = os.path.join(nsp3_output_dir, "features.csv")
    predictions_csv_path = os.path.join(args.output, "results.csv")
    xgb_model_path = "best_xgb_model.pkl"
 
    start = time.time()
    try:
        print("\nRunning NetSurfP-3.0...")
        run_nsp3(args.input, nsp3_output_dir, nsp3_model_path)

        print("\nComputing features...")
        compute_features(nsp_result_path, features_csv_path)

        print("\nRunning crystallizability predictions...") 
        make_predictions(features_csv_path, xgb_model_path, predictions_csv_path, args.input)

        print("\nSorting the results...") 
        postprocess_predictions(predictions_csv_path, args.input, predictions_csv_path)

    finally:
        print("\nCleaning up temporary files...") 
        clean_up(nsp3_output_dir)
        pass
    
    print(f"\nFinished in {time.time() - start:.2f} seconds.")


if __name__ == "__main__":
    main()