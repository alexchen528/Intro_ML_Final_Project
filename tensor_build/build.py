import os
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from helper import get_muon_energy, build_event_image, dom_index

DATA_DIR = "/ceph/work/SATORI/alex/sim_new/sim_3/output/hits"
PATTERN  = "photon_*.parquet"
OUT_DIR  = "/ceph/work/SATORI/alex/intro_ML/chunks"
os.makedirs(OUT_DIR, exist_ok = True)


def main(chunk_idx, files_per_chunk): #parallize by chuncks

    all_files = sorted(glob.glob(os.path.join(DATA_DIR, PATTERN)))
    n_files = len(all_files)


    start = chunk_idx * files_per_chunk
    end   = min(start + files_per_chunk, n_files)
    if start >= n_files:
        return

    files = all_files[start:end]

    X_list = []
    y_list = []

    n_total_events = 0
    n_used = 0
    n_skip_muon = 0
    n_skip_hits = 0

    for f in tqdm(files, desc = f"Chunk {chunk_idx}"):
        df = pd.read_parquet(f, columns=["mc_truth", "photons"])
        for mc_truth, photons in zip(df["mc_truth"], df["photons"]):
            n_total_events += 1

            muon_E = get_muon_energy(mc_truth)
            if muon_E is None or muon_E <= 0:
                n_skip_muon += 1
                continue
            y = np.log10(muon_E)

            img = build_event_image(photons)
            if img is None:
                n_skip_hits += 1
                continue

            img = img[np.newaxis, :, :]  # (1,44,128)
            X_list.append(img)
            y_list.append(y)
            n_used += 1


    print("Total events seen:", n_total_events)
    print("Events used:", n_used)
    print("Skipped (no muon):", n_skip_muon)
    print("Skipped (no hits / outside window):", n_skip_hits)

   

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)

    out_X = os.path.join(OUT_DIR, f"X_chunk_{chunk_idx:03d}.npy")
    out_y = os.path.join(OUT_DIR, f"y_chunk_{chunk_idx:03d}.npy")

    np.save(out_X, X)
    np.save(out_y, y)
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-idx", type=int, required=True)
    parser.add_argument("--files-per-chunk", type=int, default=10)
    args = parser.parse_args()
    main(args.chunk_idx, args.files_per_chunk)