import os
import glob
import time
import pickle
import random
import logging
import argparse

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, cohen_kappa_score
from sklearn.svm import SVC

# -------------------------
# 1) Argument parser
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Directory containing .pkl files')
    # parser.add_argument('--config', required=True, help='Path to dataset-specific config file')
    parser.add_argument('--n_trials', type=int, default=40, help='Number of Optuna trials') 
    parser.add_argument('--output', default='results', help='Base filename for outputs')
    return parser.parse_args()

# -------------------------
# 2) Utilities
# -------------------------
def load_subject(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['data'], data['label']

def get_channel_features(data, ch_idx):
    # data shape: (n_trials, n_chan, n_token, feat_dim)
    data_ch = data[:, ch_idx, :, :]
    n_trials, n_token, feat_dim = data_ch.shape
    return data_ch.reshape(n_trials, n_token * feat_dim)

# -------------------------
# 3) Main
# -------------------------
def main():
    args = parse_args()

    # Import dataset-specific config as module
    # spec = __import__(os.path.splitext(os.path.basename(args.config))[0])
    # CHANNELS = spec.CHANNELS
    # SUBJECTS = spec.SUBJECTS

    SUBJECTS = ['sub-B61', 'sub-B62', 'sub-B63', 'sub-B64', 'sub-B65', 'sub-B66', 'sub-B67', 
                'sub-B68', 'sub-B69', 'sub-B70', 'sub-B71', 'sub-B72', 'sub-B73', 'sub-B74', 
                'sub-B75', 'sub-B76', 'sub-B77', 'sub-B78', 'sub-B79', 'sub-B80', 'sub-B81']
    
    CHANNELS = ['Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'C1', 'C3', 'C5', 'C2', 'C4', 'C6', 'F4', 
                'FC2', 'FC4', 'FC6', 'CP2', 'CP4', 'CP6', 'P4', 'F3', 'FC1', 'FC3', 'FC5', 
                'CP1', 'CP3', 'CP5', 'P3']

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    random_state = 2023
    np.random.seed(random_state)
    random.seed(random_state)

    # Define hyperparameter grids for SVM
    C_space = np.logspace(-1, 2, 20).tolist()
    gamma_space = ['scale', 'auto']

    subject_files = sorted(glob.glob(os.path.join(args.data_dir, '*.pkl')))

    results_pickle = f'{args.output}_optuna_channel_results.pkl'
    log_txt = f'{args.output}_channel_results.txt'

    if os.path.exists(results_pickle):
        with open(results_pickle, 'rb') as f:
            all_results = pickle.load(f)
    else:
        all_results = []

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    start_time = time.time()

    for ch_idx, ch_name in enumerate(CHANNELS):
        logging.info(f"Starting channel: {ch_name} (index {ch_idx})")

        for test_path in subject_files:
            subj = os.path.basename(test_path)
            subj_id = os.path.splitext(subj)[0]
            if subj_id not in SUBJECTS:
                continue

            if any(r['channel'] == ch_name and r['subject'] == subj_id for r in all_results):
                continue

            # Load test subject
            data_test, label_test = load_subject(test_path)
            X_test = get_channel_features(data_test, ch_idx)
            y_test = label_test

            # Build training set from all other subjects
            X_train_list, y_train_list = [], []
            for tp in subject_files:
                sid = os.path.splitext(os.path.basename(tp))[0]
                if sid == subj_id:
                    continue
                d, l = load_subject(tp)
                Xt_ch = get_channel_features(d, ch_idx)
                X_train_list.append(Xt_ch)
                y_train_list.append(l)
            X_train_full = np.vstack(X_train_list)
            y_train_full = np.hstack(y_train_list)

            # Shuffle full training set
            perm = np.arange(X_train_full.shape[0])
            np.random.shuffle(perm)
            X_train_full = X_train_full[perm]
            y_train_full = y_train_full[perm]

            # Objective: single 80/20 split for hyperparameter search
            def objective(trial):
                C_val = trial.suggest_categorical('C', C_space)
                gamma_val = trial.suggest_categorical('gamma', gamma_space)

                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train_full, y_train_full,
                    test_size=0.2, stratify=y_train_full, random_state=random_state
                )

                clf = SVC(
                    C=C_val,
                    kernel='rbf',
                    gamma=gamma_val,
                    random_state=random_state
                )
                clf.fit(X_tr, y_tr)
                y_pred_val = clf.predict(X_val)
                return f1_score(y_val, y_pred_val, average='macro')

            sampler = optuna.samplers.TPESampler(seed=random_state)
            study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
            study.optimize(objective, n_trials=args.n_trials, n_jobs=-1)

            best_params = study.best_params
            best_inner_f1 = study.best_value

            best_svc = SVC(
                C=best_params['C'],
                kernel='rbf',
                gamma=best_params['gamma'],
                random_state=random_state
            )
            best_svc.fit(X_train_full, y_train_full)

            # Evaluate on held-out test set
            y_pred_test = best_svc.predict(X_test)
            test_acc       = accuracy_score(y_test, y_pred_test)
            test_f1        = f1_score(y_test, y_pred_test, average='macro')
            test_precision = precision_score(y_test, y_pred_test, average='macro')
            test_kappa     = cohen_kappa_score(y_test, y_pred_test)

            result = {
                'channel':           ch_name,
                'subject':           subj_id,
                'best_params':       best_params,
                'best_inner_f1':     best_inner_f1,
                'test_accuracy':     test_acc,
                'test_f1_macro':     test_f1,
                'test_precision_macro': test_precision,
                'test_kappa':        test_kappa
            }
            all_results.append(result)

            with open(results_pickle, 'wb') as f:
                pickle.dump(all_results, f)
            with open(log_txt, 'a') as f:
                f.write(
                    f"{ch_name}\t{subj_id}\t"
                    f"acc={test_acc:.4f}\t"
                    f"f1={test_f1:.4f}\t"
                    f"precision={test_precision:.4f}\t"
                    f"kappa={test_kappa:.4f}\t"
                    f"inner_f1={best_inner_f1:.4f}\n"
                )
            logging.info(
                f"Channel: {ch_name}, Subject: {subj_id} → "
                f"acc: {test_acc:.4f}, f1: {test_f1:.4f}, "
                f"precision: {test_precision:.4f}, kappa: {test_kappa:.4f}, "
                f"inner_f1: {best_inner_f1:.4f}"
            )

    end_time = time.time()
    logging.info(f"Total time: {(end_time - start_time) / 60:.2f} minutes")


if __name__ == '__main__':
    main()
