import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import ROOT, build_features

MODEL_DIR = ROOT / "results" / "model"


def load_model():
    with open(MODEL_DIR / "my_own_model.pkl", "rb") as f:
        art = pickle.load(f)
    return art["model"], art["features"]


def main():
    model, feats = load_model()
    train_df, test_df = build_features()

    ids = pd.read_csv(MODEL_DIR / "holdout_ids.csv").SK_ID_CURR
    ho = train_df[train_df.SK_ID_CURR.isin(ids)]
    auc = roc_auc_score(ho.TARGET, model.predict_proba(ho[feats])[:, 1])
    print(f"AUC on test set: {auc:.2f}")

    sub = pd.DataFrame({"SK_ID_CURR": test_df.SK_ID_CURR,
                        "TARGET": model.predict_proba(test_df[feats])[:, 1]})
    sub.to_csv(ROOT / "results" / "submission.csv", index=False)


if __name__ == "__main__":
    main()
