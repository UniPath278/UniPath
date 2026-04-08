"""
Logistic Regression for Slide-level Classification

This script performs slide-level classification using pre-extracted features
stored in HDF5 files. It follows the evaluation protocol from the TITAN paper:
logistic regression with grid search over regularization strength, balanced
class weighting, and bootstrap confidence intervals.

Usage:
    Edit DATASET_CONFIGS below, then run:
        python slide_logistic_regression.py
"""

import os
import traceback
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import h5py
import joblib
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Default hyper-parameters (TITAN paper settings)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "c_min": 1e-6,
    "c_max": 1e5,
    "c_num": 45,
    "max_iter": 1000,
    "solver": "lbfgs",
    "use_standardization": True,
    "bootstrap_samples": 1000,
    "random_seed": 42,
}

# ---------------------------------------------------------------------------
# Batch experiment configurations
# Add / uncomment entries here to run multiple experiments in sequence.
# ---------------------------------------------------------------------------
DATASET_CONFIGS = [
    {
        "dataset_name": "EBRAINS",
        "h5_folder": "EBRAINS_feature/slide_features_unipath",
        "dataset_folder": "csv",
        "output_folder": "./ebrains30_res",
        "csv_prefix": "ebrains30",
        "id_column": "image_id",
        "label_column": "label",
    },
    {
        "dataset_name": "EBRAINS",
        "h5_folder": "EBRAINS_feature/slide_features_titan",
        "dataset_folder": "csv",
        "output_folder": "./ebrains30_res",
        "csv_prefix": "ebrains30",
        "id_column": "image_id",
        "label_column": "label",
    },
    {
        "dataset_name": "EBRAINS",
        "h5_folder": "EBRAINS_feature/slide_features_chief",
        "dataset_folder": "csv",
        "output_folder": "./ebrains30_res",
        "csv_prefix": "ebrains30",
        "id_column": "image_id",
        "label_column": "label",
    },
    {
        "dataset_name": "EBRAINS",
        "h5_folder": "EBRAINS_feature/slide_features_gigapath",
        "dataset_folder": "csv",
        "output_folder": "./ebrains30_res",
        "csv_prefix": "ebrains30",
        "id_column": "image_id",
        "label_column": "label",
    },
]


# ===================================================================
# Core experiment class
# ===================================================================
class SlideLogisticRegression:
    """Logistic Regression experiment for slide-level classification."""

    def __init__(self, config: dict):
        self.config = {**DEFAULT_PARAMS, **config}
        self.scaler = None
        self.label_encoder = None
        self.best_model = None
        self.h5_folder_name = os.path.basename(
            self.config["h5_folder"].rstrip("/\\")
        )
        self._pooling_logged = False

        np.random.seed(self.config["random_seed"])

    # ----------------------------------------------------------------
    # Data loading
    # ----------------------------------------------------------------
    @staticmethod
    def _load_h5_features(h5_path: str) -> np.ndarray:
        """Load feature vector from an HDF5 file; apply mean-pooling if multi-dimensional."""
        with h5py.File(h5_path, "r") as f:
            features = f["features"][:]
        if features.ndim > 1:
            features = np.mean(features, axis=0)
        return features

    def _load_splits(self) -> dict[str, pd.DataFrame]:
        """Load train / val / test CSV splits."""
        splits: dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            csv_path = os.path.join(
                self.config["dataset_folder"],
                f"{self.config['csv_prefix']}_{split}.csv",
            )
            if os.path.exists(csv_path):
                splits[split] = pd.read_csv(csv_path)

        if "val" not in splits and "train" in splits:
            train_df = splits["train"]
            val_size = int(0.2 * len(train_df))
            splits["val"] = train_df.tail(val_size).copy()
            splits["train"] = train_df.head(len(train_df) - val_size).copy()

        return splits

    def _encode_labels(self, splits: dict[str, pd.DataFrame]):
        """Fit a LabelEncoder on all splits and add an `encoded_label` column."""
        all_labels = np.concatenate(
            [df[self.config["label_column"]].unique() for df in splits.values()]
        )
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(all_labels)
        for df in splits.values():
            df["encoded_label"] = self.label_encoder.transform(
                df[self.config["label_column"]]
            )

    def _load_features(
        self, splits: dict[str, pd.DataFrame]
    ) -> dict[str, dict]:
        """Read HDF5 feature files for every sample in each split."""
        features_data = {}
        for split_name, df in splits.items():
            feat_list, id_list, label_list = [], [], []
            missing = 0
            for _, row in tqdm(
                df.iterrows(), total=len(df), desc=f"Loading {split_name}"
            ):
                image_id = row[self.config["id_column"]]
                h5_path = os.path.join(
                    self.config["h5_folder"], f"{image_id}.h5"
                )
                if os.path.exists(h5_path):
                    feat = self._load_h5_features(h5_path)
                    if feat.ndim > 1 and not self._pooling_logged:
                        print(
                            f"[INFO] Multi-dim features detected; "
                            f"applying mean-pooling -> {feat.shape}"
                        )
                        self._pooling_logged = True
                    feat_list.append(feat)
                    id_list.append(image_id)
                    if "encoded_label" in row.index:
                        label_list.append(row["encoded_label"])
                else:
                    missing += 1
            if missing:
                print(
                    f"[WARN] {split_name}: {missing}/{len(df)} HDF5 files missing"
                )
            if feat_list:
                features_data[split_name] = {
                    "features": np.array(feat_list),
                    "labels": np.array(label_list) if label_list else None,
                    "ids": id_list,
                }
        return features_data

    # ----------------------------------------------------------------
    # Training & evaluation
    # ----------------------------------------------------------------
    @staticmethod
    def _compute_auc(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
        """Compute AUC metrics (binary and multi-class OvR / OvO)."""
        metrics: dict[str, float] = {}
        n_classes = len(np.unique(y_true))
        unique = np.unique(y_true)
        if not np.array_equal(unique, np.arange(n_classes)):
            mapping = {old: new for new, old in enumerate(unique)}
            y_true = np.array([mapping[y] for y in y_true])

        try:
            if n_classes == 2:
                val = roc_auc_score(y_true, y_proba[:, 1])
                metrics.update(auc=val, auc_ovr=val, auc_ovo=val)
            else:
                labels = np.arange(n_classes)
                for mode in ("ovr", "ovo"):
                    try:
                        metrics[f"auc_{mode}"] = roc_auc_score(
                            y_true,
                            y_proba,
                            multi_class=mode,
                            average="macro",
                            labels=labels,
                        )
                    except Exception:
                        metrics[f"auc_{mode}"] = np.nan
                metrics["auc"] = metrics.get("auc_ovr", np.nan)
        except Exception:
            metrics = {"auc": np.nan, "auc_ovr": np.nan, "auc_ovo": np.nan}
        return metrics

    def _train_and_evaluate(self, features_data: dict) -> tuple:
        """Grid-search over C on val set, evaluate best model on test set."""
        X_train = features_data["train"]["features"]
        y_train = features_data["train"]["labels"]
        X_val = features_data["val"]["features"]
        y_val = features_data["val"]["labels"]
        X_test = features_data["test"]["features"]
        y_test = features_data["test"]["labels"]

        # Handle NaN
        for arr_name, arr in [
            ("X_train", X_train),
            ("X_val", X_val),
            ("X_test", X_test),
        ]:
            nan_count = np.isnan(arr).sum()
            if nan_count:
                print(f"[WARN] {arr_name} contains {nan_count} NaN values; replacing with 0")
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        # Optional standardization
        if self.config["use_standardization"]:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)

        # Grid search
        c_values = np.logspace(
            np.log10(self.config["c_min"]),
            np.log10(self.config["c_max"]),
            num=self.config["c_num"],
        )

        best_score, best_c = 0.0, None
        for c in tqdm(c_values, desc="Grid search"):
            clf = LogisticRegression(
                C=c,
                max_iter=self.config["max_iter"],
                solver=self.config["solver"],
                multi_class="multinomial",
                random_state=self.config["random_seed"],
                class_weight="balanced",
                n_jobs=-1,
            )
            try:
                clf.fit(X_train, y_train)
                score = balanced_accuracy_score(y_val, clf.predict(X_val))
                if score > best_score:
                    best_score, best_c, self.best_model = score, c, clf
            except Exception as e:
                print(f"[WARN] Training failed for C={c:.2e}: {e}")

        if self.best_model is None:
            raise RuntimeError("No valid model found during grid search")

        # Test evaluation
        y_pred = self.best_model.predict(X_test)
        y_proba = self.best_model.predict_proba(X_test)

        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        }
        results.update(self._compute_auc(y_test, y_proba))

        outputs = {
            "y_true": y_test,
            "y_pred": y_pred,
            "y_pred_proba": y_proba,
            "test_ids": features_data["test"]["ids"],
        }
        return results, outputs, best_c

    # ----------------------------------------------------------------
    # Bootstrap confidence intervals
    # ----------------------------------------------------------------
    def _bootstrap(
        self, outputs: dict, n: int = 1000, alpha: float = 0.95
    ) -> tuple[dict, dict]:
        metrics_keys = [
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "f1_weighted",
            "auc",
            "auc_ovr",
            "auc_ovo",
        ]
        boot: dict[str, list] = {k: [] for k in metrics_keys}

        y_true = outputs["y_true"]
        y_pred = outputs["y_pred"]
        y_proba = outputs["y_pred_proba"]
        n_samples = len(y_true)

        for _ in tqdm(range(n), desc="Bootstrap"):
            idx = np.random.choice(n_samples, n_samples, replace=True)
            yt, yp, ypp = y_true[idx], y_pred[idx], y_proba[idx]

            boot["accuracy"].append(accuracy_score(yt, yp))
            boot["balanced_accuracy"].append(balanced_accuracy_score(yt, yp))
            boot["f1_macro"].append(f1_score(yt, yp, average="macro"))
            boot["f1_weighted"].append(f1_score(yt, yp, average="weighted"))

            try:
                auc_m = self._compute_auc(yt, ypp)
                for k in ("auc", "auc_ovr", "auc_ovo"):
                    boot[k].append(auc_m.get(k, np.nan))
            except Exception:
                for k in ("auc", "auc_ovr", "auc_ovo"):
                    boot[k].append(np.nan)

        means, stds = {}, {}
        lo_pct = (1 - alpha) / 2 * 100
        hi_pct = (1 + alpha) / 2 * 100
        for k, vals in boot.items():
            arr = np.array(vals)
            arr = arr[~np.isnan(arr)]
            if len(arr):
                means[k] = float(np.mean(arr))
                stds[k] = float(np.std(arr))
                means[f"{k}_ci_lower"] = float(np.percentile(arr, lo_pct))
                means[f"{k}_ci_upper"] = float(np.percentile(arr, hi_pct))
            else:
                means[k] = stds[k] = np.nan
        return means, stds

    # ----------------------------------------------------------------
    # Saving results
    # ----------------------------------------------------------------
    def _save_results(
        self,
        results: dict,
        means: dict,
        stds: dict,
        outputs: dict,
        best_c: float,
    ) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.config["output_folder"]
        tag = f"{self.config['dataset_name']}_{self.h5_folder_name}"

        # Model
        joblib.dump(
            {
                "model": self.best_model,
                "scaler": self.scaler,
                "label_encoder": self.label_encoder,
                "best_C": best_c,
                "config": self.config,
            },
            os.path.join(out, f"best_model_{tag}_{ts}.pkl"),
        )

        # Predictions
        pred_df = pd.DataFrame(
            {
                "id": outputs["test_ids"],
                "true_label": self.label_encoder.inverse_transform(
                    outputs["y_true"]
                ),
                "predicted_label": self.label_encoder.inverse_transform(
                    outputs["y_pred"]
                ),
            }
        )
        for i, cls_name in enumerate(self.label_encoder.classes_):
            pred_df[f"prob_{cls_name}"] = outputs["y_pred_proba"][:, i]
        pred_df["correct"] = pred_df["true_label"] == pred_df["predicted_label"]
        pred_df["confidence"] = np.max(outputs["y_pred_proba"], axis=1)
        pred_df.to_csv(os.path.join(out, f"predictions_{tag}_{ts}.csv"), index=False)

        # Summary
        summary = {
            "timestamp": ts,
            "dataset_name": self.config["dataset_name"],
            "h5_folder": self.h5_folder_name,
            "num_test_samples": len(outputs["y_true"]),
            "num_classes": len(self.label_encoder.classes_),
            "feature_dim": self.best_model.coef_.shape[1],
            "best_C": best_c,
            "solver": self.config["solver"],
            "max_iter": self.config["max_iter"],
            "bootstrap_samples": self.config["bootstrap_samples"],
        }
        for k, v in results.items():
            summary[k] = v
        for k, v in means.items():
            if not k.endswith(("_ci_lower", "_ci_upper")):
                summary[f"{k}_bootstrap_mean"] = v
                summary[f"{k}_bootstrap_std"] = stds.get(k, np.nan)
            else:
                summary[k] = v
        pd.DataFrame([summary]).to_csv(
            os.path.join(out, f"summary_{tag}_{ts}.csv"), index=False
        )

        # Confusion matrix
        cm = confusion_matrix(outputs["y_true"], outputs["y_pred"])
        classes = self.label_encoder.classes_
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        cm_df.to_csv(os.path.join(out, f"confusion_matrix_{tag}_{ts}.csv"))

        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        pd.DataFrame(cm_norm, index=classes, columns=classes).to_csv(
            os.path.join(out, f"confusion_matrix_norm_{tag}_{ts}.csv")
        )

        # Per-class metrics
        prec, rec, f1, sup = precision_recall_fscore_support(
            outputs["y_true"], outputs["y_pred"], average=None
        )
        pc = pd.DataFrame(
            {
                "class": classes,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": sup,
            }
        )
        if len(classes) > 2:
            y_bin = label_binarize(
                outputs["y_true"], classes=range(len(classes))
            )
            aucs = []
            for i in range(len(classes)):
                try:
                    if len(np.unique(y_bin[:, i])) > 1:
                        aucs.append(
                            roc_auc_score(
                                y_bin[:, i], outputs["y_pred_proba"][:, i]
                            )
                        )
                    else:
                        aucs.append(np.nan)
                except Exception:
                    aucs.append(np.nan)
            pc["auc"] = aucs
        pc.to_csv(
            os.path.join(out, f"per_class_metrics_{tag}_{ts}.csv"),
            index=False,
        )

        # Label mapping
        mapping_path = os.path.join(out, f"label_mapping_{tag}_{ts}.csv")
        pd.DataFrame(
            {
                "original_label": self.label_encoder.classes_,
                "encoded_label": self.label_encoder.transform(
                    self.label_encoder.classes_
                ),
            }
        ).to_csv(mapping_path, index=False)


        return ts

    # ----------------------------------------------------------------
    # Main entry point
    # ----------------------------------------------------------------
    def run(self):
        """Run the full experiment pipeline."""
        os.makedirs(self.config["output_folder"], exist_ok=True)

        print("=" * 60)
        print(f"Dataset : {self.config['dataset_name']}")
        print(f"Features: {self.config['h5_folder']}")
        print("=" * 60)

        # Load & encode
        splits = self._load_splits()
        if not splits:
            raise FileNotFoundError("No CSV split files found")
        for name, df in splits.items():
            print(f"  {name}: {len(df)} samples")

        # Show label distribution
        label_col = self.config["label_column"]
        print(f"\nLabel distribution ({label_col}):")
        counts_list = []
        for split_name, df in splits.items():
            if label_col in df.columns:
                counts_list.append(
                    df[label_col].value_counts().rename(split_name)
                )
        if counts_list:
            label_table = pd.concat(counts_list, axis=1).fillna(0).astype(int)
            label_table["total"] = label_table.sum(axis=1)
            label_table = label_table.sort_values("total", ascending=False)
            print(label_table)

        self._encode_labels(splits)
        features_data = self._load_features(splits)

        for required in ("train", "val", "test"):
            if required not in features_data:
                raise RuntimeError(f"Missing required split: {required}")

        # Train
        results, outputs, best_c = self._train_and_evaluate(features_data)

        # Bootstrap
        means, stds = self._bootstrap(
            outputs, n=self.config["bootstrap_samples"]
        )

        # Save
        ts = self._save_results(results, means, stds, outputs, best_c)

        # Print summary
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        print(f"  H5 folder        : {self.h5_folder_name}")
        print(
            f"  Balanced Accuracy: {results['balanced_accuracy']:.3f}"
            f" ± {stds['balanced_accuracy']:.4f}"
        )
        print(
            f"  Weighted F1      : {results['f1_weighted']:.3f}"
            f" ± {stds['f1_weighted']:.4f}"
        )
        if not np.isnan(results.get("auc", np.nan)):
            print(
                f"  AUC              : {results['auc']:.3f}"
                f" ± {stds.get('auc', 0):.4f}"
            )
        print(f"  Outputs saved to : {self.config['output_folder']}")
        print("=" * 60)

        return {
            "h5_folder": self.h5_folder_name,
            "balanced_accuracy": results["balanced_accuracy"],
            "balanced_accuracy_std": stds["balanced_accuracy"],
            "f1_weighted": results["f1_weighted"],
            "f1_weighted_std": stds["f1_weighted"],
            "auc": results.get("auc", np.nan),
            "auc_std": stds.get("auc", np.nan),
        }


# ===================================================================
# Batch runner
# ===================================================================
def run_batch(configs: list[dict]):
    """Run multiple experiments in sequence and print a final summary."""
    print("=" * 60)
    print(f"Batch run: {len(configs)} experiment(s)")
    print("=" * 60)

    all_results = []

    for i, cfg in enumerate(configs, 1):
        tag = cfg.get("dataset_name", f"experiment_{i}")
        h5_tag = os.path.basename(cfg.get("h5_folder", "").rstrip("/\\"))
        print(f"\n>>> [{i}/{len(configs)}] {tag} | {h5_tag}")
        try:
            exp = SlideLogisticRegression(cfg)
            res = exp.run()
            all_results.append({"dataset": tag, "status": "OK", **res})
        except Exception as e:
            print(f"[ERROR] {tag}: {e}")
            traceback.print_exc()
            all_results.append({
                "dataset": tag,
                "status": f"FAILED",
                "h5_folder": h5_tag,
                "balanced_accuracy": np.nan,
                "balanced_accuracy_std": np.nan,
                "f1_weighted": np.nan,
                "f1_weighted_std": np.nan,
                "auc": np.nan,
                "auc_std": np.nan,
            })

    # Final summary table
    print("\n\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    header = f"{'Dataset':<15} {'H5 Features':<45} {'ACC':>14} {'F1':>14} {'AUC':>14}"
    print(header)
    print("-" * 100)
    for r in all_results:
        if r["status"] == "OK":
            acc_str = f"{r['balanced_accuracy']:.3f}±{r['balanced_accuracy_std']:.4f}"
            f1_str = f"{r['f1_weighted']:.3f}±{r['f1_weighted_std']:.4f}"
            if not np.isnan(r["auc"]):
                auc_str = f"{r['auc']:.3f}±{r['auc_std']:.4f}"
            else:
                auc_str = "N/A"
        else:
            acc_str = f1_str = auc_str = "FAILED"
        print(f"{r['dataset']:<15} {r['h5_folder']:<45} {acc_str:>14} {f1_str:>14} {auc_str:>14}")
    print("=" * 100)


# ===================================================================
# Entry point
# ===================================================================
if __name__ == "__main__":
    run_batch(DATASET_CONFIGS)