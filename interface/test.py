# shrink_xy_mat_script.py
# Réduit un fichier .mat (HDF5 v7.3) contenant X (features x samples) et Y (1 x samples)
# en ne gardant qu'une fraction stratifiée par classe. Écrit un nouveau .h5.

import os, math
import numpy as np
import h5py

# ========= PARAMÈTRES À ÉDITER =========
IN_PATH      = r"C:\Users\Usuario\Documents\GitHub\Hypertool\identification\data/XY_train_background.mat"   # chemin d'entrée
OUT_PATH     = None   # None => génère automatiquement "<nom>_10pct.h5" dans le même dossier
FRAC_LARGE         = 0.10   # fraction par classe (ex: 0.10 = 10%)
KEEP_ALL_UNDER = 100_000  # seuil: si n_classe < KEEP_ALL_UNDER -> on garde tout
SEED         = 42     # seed pour reproductibilité
BATCH_COLS   = 4096   # taille des lots de colonnes copiées (adapter à ta RAM/SSD)
COMPRESSION  = "gzip" # "gzip" (classique) ou "lzf" (plus rapide, moins compressé) ou None
COMP_LEVEL   = 4      # 1..9 (ignoré si COMPRESSION="lzf")
MIN_PER_CLASS = 1     # minimum d'échantillons par classe à conserver
SAVE_SOURCE_INDICES = True  # enregistre les indices des samples conservés
# =======================================


def stratified_indices(y, frac_large=0.10, keep_all_under=100_000,
                                 seed=42, min_per_class=1):
    rng = np.random.RandomState(seed)
    idx_all = []
    classes, counts = np.unique(y, return_counts=True)
    for c, n in zip(classes, counts):
        idx_c = np.where(y == c)[0]
        if n < keep_all_under:
            pick = idx_c  # on garde tout
        else:
            k = max(min_per_class, int(math.floor(frac_large * n)))
            k = min(k, n)
            pick = rng.choice(idx_c, size=k, replace=False)
        idx_all.append(pick)
    idx_all = np.concatenate(idx_all)
    rng.shuffle(idx_all)  # mélange global (entre classes)
    return idx_all, classes, counts

def main(in_path, out_path, frac, seed, batch_cols, compression, comp_level, min_per_class):
    if not (0 < frac <= 1.0):
        raise ValueError("FRAC doit être dans l'intervalle (0, 1].")

    if out_path is None:
        base = os.path.splitext(os.path.basename(in_path))[0]
        out_path = os.path.join(os.path.dirname(os.path.abspath(in_path)),
                                f"{base}_{int(frac * 100)}pct.h5")

    if os.path.abspath(in_path) == os.path.abspath(out_path):
        raise ValueError("OUT_PATH ne doit pas écraser le fichier d'entrée.")

    with h5py.File(in_path, "r") as fin:
        # --- lire Y (souvent (1, N) dans les .mat HDF5 v7.3)
        Y_dset = fin["Y"]
        if len(Y_dset.shape) == 2 and Y_dset.shape[0] == 1:
            y = np.array(Y_dset[0, :])
        else:
            y = np.array(Y_dset[:]).squeeze()
        y = y.astype(np.int64, copy=False)

        X_dset = fin["X"]  # attendu: (n_features, n_samples)
        if len(X_dset.shape) != 2:
            raise ValueError(f"X a une forme inattendue: {X_dset.shape} (attendu 2D: features x samples)")
        n_features, n_samples = X_dset.shape

        idx_keep, classes, counts_src = stratified_indices(
            y,
            frac_large=FRAC_LARGE,
            keep_all_under=KEEP_ALL_UNDER,
            seed=SEED,
            min_per_class=MIN_PER_CLASS,
        )
        n_keep = idx_keep.size

        # OUT_PATH par défaut
        if out_path is None:
            base = os.path.splitext(os.path.basename(in_path))[0]
            out_path = os.path.join(os.path.dirname(os.path.abspath(in_path)),
                                    f"{base}_{int(frac*100)}pct.h5")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        # --- écriture du nouveau fichier HDF5
        with h5py.File(out_path, "w") as fout:
            # dataset X (même orientation que la source)
            X_small = fout.create_dataset(
                "X",
                shape=(n_features, n_keep),
                dtype=X_dset.dtype,
                chunks=(min(n_features, 2048), min(n_keep, 64)),
                compression=compression,
                compression_opts=(comp_level if compression == "gzip" else None),
                shuffle=True,
            )
            # dataset Y (forme (1, n_keep) pour compat MATLAB)
            Y_small = fout.create_dataset(
                "Y",
                shape=(1, n_keep),
                dtype=y.dtype,
                chunks=(1, min(n_keep, 8192)),
                compression=compression,
                compression_opts=(comp_level if compression == "gzip" else None),
                shuffle=True,
            )

            # métadonnées
            fout.attrs["source_file"] = os.path.basename(in_path)
            fout.attrs["sampling_frac"] = frac
            fout.attrs["seed"] = seed
            fout.attrs["note"] = "Stratified subsample per class; orientation = (features x samples)."

            if SAVE_SOURCE_INDICES:
                fout.create_dataset(
                    "indices_source",
                    data=idx_keep,
                    compression=compression,
                    compression_opts=(comp_level if compression == "gzip" else None),
                    shuffle=True,
                )

            # --- copie en lots (meilleure I/O)
            # On trie les colonnes dans chaque lot pour accès séquentiel
            for start in range(0, n_keep, BATCH_COLS):
                end = min(start + BATCH_COLS, n_keep)
                cols = idx_keep[start:end]  # ordre global voulu
                cols_sorted = np.sort(cols)  # pour accès disque séquentiel
                # lire en ordre trié
                X_batch_sorted = X_dset[:, cols_sorted]  # (features, batch)
                y_batch_sorted = y[cols_sorted]  # (batch,)
                # remettre dans l'ordre original 'cols'
                pos_in_sorted = np.searchsorted(cols_sorted, cols)
                X_batch = X_batch_sorted[:, pos_in_sorted]
                y_batch = y_batch_sorted[pos_in_sorted]
                # écrire
                X_small[:, start:end] = X_batch
                Y_small[0, start:end] = y_batch

        # --- récapitulatif
        mask_keep = np.zeros(n_samples, dtype=bool)
        mask_keep[idx_keep] = True
        y_keep = y[mask_keep]
        counts_dst = [int(np.sum(y_keep == c)) for c in classes]

        print(f"✅ Terminé : {out_path}")
        print(f"X: {n_features} features, {n_samples} → {n_keep} samples (FRAC={frac})")
        print("Comptes par classe (source → sortie) :")
        for c, ns, nd in zip(classes, counts_src, counts_dst):
            print(f"  classe {c}: {int(ns)} → {int(nd)}")

if __name__ == "__main__":
    # Appel direct quand tu lances depuis l’IDE
    # (Tu peux aussi dupliquer ce bloc pour tester plusieurs FRAC avec différents OUT_PATH)
    main(
        in_path=IN_PATH,
        out_path=OUT_PATH,
        frac=FRAC_LARGE,
        seed=SEED,
        batch_cols=BATCH_COLS,
        compression=COMPRESSION,
        comp_level=COMP_LEVEL,
        min_per_class=MIN_PER_CLASS,
    )
