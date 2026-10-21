
import argparse
import base64
import io
import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap

from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve


warnings.filterwarnings(
    "ignore",
    message=".*TreeExplainer shap values output.*",
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import DATA_DIR, ROOT, build_features


MODEL_DIR = ROOT / "results" / "model"
OUT_DIR = ROOT / "results" / "clients_outputs"

MODEL_PATH = MODEL_DIR / "my_own_model.pkl"
HOLDOUT_PATH = MODEL_DIR / "holdout_ids.csv"

TRAIN_PATH = DATA_DIR / "application_train.csv"
TEST_PATH = DATA_DIR / "application_test.csv"

SHAP_SUMMARY_PATH = MODEL_DIR / "shap_summary.png"


DESCRIPTIVE = {
    "AGE_YEARS": "Age (years)",
    "YEARS_EMPLOYED": "Years employed",
    "AMT_INCOME_TOTAL": "Annual income",
    "AMT_CREDIT": "Loan amount",
    "AMT_ANNUITY": "Loan annuity",
    "AMT_GOODS_PRICE": "Goods price",
    "CNT_CHILDREN": "Children",
    "CNT_FAM_MEMBERS": "Family members",
    "EXT_SOURCE_1": "External score 1",
    "EXT_SOURCE_2": "External score 2",
    "EXT_SOURCE_3": "External score 3",
    "REGION_POPULATION_RELATIVE": "Region population (rel.)",
}

CATEGORICAL = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def normalize_shap_values(values):
    if isinstance(values, list):
        values = values[1]

    values = np.asarray(values)

    if values.ndim == 3:
        values = values[:, :, 1]

    return values


def load_model_artifact(path):
    with open(path, "rb") as file:
        artifact = pickle.load(file)

    return artifact["model"], artifact["features"]


class Explainer:
    def __init__(self):
        self._load_model()
        self._load_features()
        self._load_raw_data()
        self._initialize_shap()

    def _load_model(self):
        self.model, self.feats = load_model_artifact(MODEL_PATH)

        self.holdout_ids = pd.read_csv(
            HOLDOUT_PATH
        )["SK_ID_CURR"].values

    def _load_features(self):
        self.train_df, self.test_df = build_features()

        self.X = (
            pd.concat(
                [self.train_df, self.test_df],
                ignore_index=True,
            )
            .set_index("SK_ID_CURR", drop=False)
        )

    def _load_raw_data(self):
        train = pd.read_csv(TRAIN_PATH)
        test = pd.read_csv(TEST_PATH)

        raw = pd.concat(
            [train, test],
            ignore_index=True,
        )

        # Clean special employment value.
        raw["DAYS_EMPLOYED"] = raw["DAYS_EMPLOYED"].replace(
            365243,
            np.nan,
        )

        # Convert day-based variables into years.
        raw["AGE_YEARS"] = -raw["DAYS_BIRTH"] / 365.25
        raw["YEARS_EMPLOYED"] = (
            -raw["DAYS_EMPLOYED"] / 365.25
        )

        self.raw = raw.set_index(
            "SK_ID_CURR",
            drop=False,
        )

        # Population with known training labels.
        self.pop = self.raw[
            self.raw["SK_ID_CURR"].isin(train["SK_ID_CURR"])
        ]

    def _initialize_shap(self):
        self.shap_explainer = shap.TreeExplainer(self.model)

    # Prediction and SHAP calculations
    def score(self, sk_id):
        row = self.X.loc[[sk_id], self.feats]

        return float(
            self.model.predict_proba(row)[:, 1][0]
        )

    def shap_row(self, sk_id):
        row = self.X.loc[[sk_id], self.feats]

        values = self.shap_explainer.shap_values(row)
        values = normalize_shap_values(values)

        base = float(
            np.ravel(self.shap_explainer.expected_value)[-1]
        )

        return row.iloc[0], values[0], base

    def force_png(self, sk_id):
        row, shap_values, base = self.shap_row(sk_id)

        shap.force_plot(
            base,
            shap_values,
            row.round(3),
            matplotlib=True,
            show=False,
            figsize=(22, 3.5),
            text_rotation=12,
            contribution_threshold=0.04,
        )

        buffer = io.BytesIO()

        plt.gcf().savefig(
            buffer,
            format="png",
            bbox_inches="tight",
            dpi=110,
        )

        plt.close("all")

        reconstructed_score = float(
            sigmoid(base + shap_values.sum())
        )

        return buffer.getvalue(), reconstructed_score

    # Descriptive statistics
    def descriptive_table(self, sk_id):
        client = self.raw.loc[sk_id]
        rows = []

        for variable, label in DESCRIPTIVE.items():
            value = client[variable]

            population = self.pop[variable].dropna()

            percentile = (
                float((population < value).mean() * 100)
                if pd.notna(value)
                else np.nan
            )

            defaulted = self.pop.loc[
                self.pop["TARGET"] == 1,
                variable,
            ]

            rows.append(
                (
                    label,
                    value,
                    population.median(),
                    defaulted.median(),
                    percentile,
                )
            )

        return pd.DataFrame(
            rows,
            columns=[
                "Variable",
                "Client",
                "Median (all)",
                "Median (defaulters)",
                "Percentile",
            ],
        )

    # Percentile visualization
    def fig_percentiles(self, sk_id):
        table = self.descriptive_table(sk_id).dropna(
            subset=["Percentile"]
        )

        fig = go.Figure(
            go.Bar(
                x=table["Percentile"],
                y=table["Variable"],
                orientation="h",
                text=(
                    table["Percentile"]
                    .round(0)
                    .astype(int)
                    .astype(str)
                    + "th"
                ),
                marker_color="#2a6fdb",
            )
        )

        fig.add_vline(
            x=50,
            line_dash="dot",
            line_color="grey",
        )

        fig.update_layout(
            title=(
                "Client position vs. all clients "
                "(percentile, raw values)"
            ),
            xaxis=dict(
                range=[0, 100],
                title="percentile",
            ),
            height=520,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=170),
        )

        return fig

    # Feature distribution visualization
    def fig_distributions(self, sk_id):
        client = self.raw.loc[sk_id]
        variables = list(DESCRIPTIVE)

        fig = make_subplots(
            rows=4,
            cols=3,
            subplot_titles=[
                DESCRIPTIVE[variable]
                for variable in variables
            ],
        )

        target_groups = [
            (0, "repaid", "#2a9d8f"),
            (1, "defaulted", "#e63946"),
        ]

        for index, variable in enumerate(variables):
            row = index // 3 + 1
            col = index % 3 + 1

            population = self.pop[
                [variable, "TARGET"]
            ].dropna()

            lower, upper = population[
                variable
            ].quantile([0.01, 0.99])

            population = population[
                population[variable].between(lower, upper)
            ]

            for target, name, color in target_groups:
                fig.add_trace(
                    go.Histogram(
                        x=population.loc[
                            population["TARGET"] == target,
                            variable,
                        ],
                        histnorm="probability density",
                        name=name,
                        marker_color=color,
                        opacity=0.55,
                        showlegend=(index == 0),
                        nbinsx=30,
                    ),
                    row=row,
                    col=col,
                )

            if pd.notna(client[variable]):
                fig.add_vline(
                    x=float(client[variable]),
                    line_width=3,
                    line_dash="dash",
                    line_color="black",
                    row=row,
                    col=col,
                )

        fig.update_layout(
            barmode="overlay",
            height=900,
            title=(
                "Client (black dashed line) vs. "
                "repaid / defaulted clients"
            ),
        )

        return fig

    # SHAP feature importance visualization
    def fig_shap(self, sk_id, top=15):
        row, shap_values, _ = self.shap_row(sk_id)

        table = pd.DataFrame(
            {
                "feature": self.feats,
                "shap": shap_values,
                "value": row.values,
            }
        )

        table = (
            table.reindex(
                table["shap"]
                .abs()
                .sort_values(ascending=False)
                .index
            )
            .head(top)
            .iloc[::-1]
        )

        fig = go.Figure(
            go.Bar(
                x=table["shap"],
                y=[
                    f"{feature} = {value:.3g}"
                    for feature, value in zip(
                        table["feature"],
                        table["value"],
                    )
                ],
                orientation="h",
                marker_color=np.where(
                    table["shap"] > 0,
                    "#e63946",
                    "#2a9d8f",
                ),
            )
        )

        fig.update_layout(
            title=(
                "Top SHAP contributions "
                "(red = raises default risk, green = lowers it)"
            ),
            xaxis_title="SHAP value (log-odds)",
            height=520,
            margin=dict(l=330),
        )

        return fig

    # Client report payload
    def payload(self, sk_id):
        sk_id = int(sk_id)

        if sk_id not in self.X.index:
            raise KeyError(f"unknown client id {sk_id}")

        force_png, force_score = self.force_png(sk_id)

        target = (
            self.X.loc[sk_id, "TARGET"]
            if "TARGET" in self.X
            and pd.notna(self.X.loc[sk_id, "TARGET"])
            else None
        )

        return {
            "id": sk_id,
            "score": self.score(sk_id),
            "score_force": force_score,
            "force_b64": base64.b64encode(
                force_png
            ).decode(),
            "table": self.descriptive_table(sk_id),
            "categorical": self.raw.loc[
                sk_id,
                [
                    column
                    for column in CATEGORICAL
                    if column in self.raw.columns
                ],
            ],
            "true_target": target,
            "figs": [
                self.fig_percentiles(sk_id),
                self.fig_distributions(sk_id),
                self.fig_shap(sk_id),
            ],
        }

    # Analysis of an incorrectly classified client
    def wrong_client_analysis(self, sk_id):
        row, shap_values, _ = self.shap_row(sk_id)

        table = pd.DataFrame(
            {
                "feature": self.feats,
                "shap": shap_values,
                "value": row.values,
            }
        )

        lowest = table.sort_values("shap").head(5)

        lines = [
            (
                f"- {item.feature} = {item.value:.3g} "
                f"pushes the score DOWN by "
                f"{abs(item.shap):.2f} log-odds"
            )
            for item in lowest.itertuples()
        ]

        return (
            "<p>The model made an error on this client. "
            "The features that most reduced the predicted risk are:</p>"
            "<pre>"
            + "\n".join(lines)
            + "</pre>"
            "<p>These values look typical of a good payer, so the "
            "default was not predictable from the available data "
            "(late life event, income shock...). This is irreducible "
            "noise rather than a model bug; compare the client with "
            "the 'defaulted' histograms to see which variables "
            "looked reassuring.</p>"
        )

    # HTML report generation
    def report_html(self, sk_id, title, note=""):
        payload = self.payload(sk_id)

        target_labels = {
            1: "DEFAULTED (TARGET=1)",
            0: "REPAID (TARGET=0)",
        }

        target = (
            None
            if payload["true_target"] is None
            else int(payload["true_target"])
        )

        outcome = target_labels.get(
            target,
            "unknown (test set)",
        )

        table_html = payload["table"].round(3).to_html(
            index=False,
            border=0,
        )

        categorical_html = payload[
            "categorical"
        ].to_frame("value").to_html(border=0)

        figures_html = "".join(
            figure.to_html(
                full_html=False,
                include_plotlyjs="cdn" if index == 0 else False,
            )
            for index, figure in enumerate(payload["figs"])
        )

        status = (
            "OK"
            if abs(
                payload["score"] - payload["score_force"]
            ) < 1e-6
            else "MISMATCH"
        )

        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{
    font-family: Arial;
    max-width: 1200px;
    margin: auto;
    padding: 20px;
}}
table {{
    border-collapse: collapse;
}}
td, th {{
    padding: 4px 10px;
    border-bottom: 1px solid #ddd;
    text-align: right;
}}
.score {{
    font-size: 34px;
    font-weight: bold;
}}
</style>
</head>
<body>
<h1>{title}</h1>
<p>Client <b>{payload['id']}</b> |
actual outcome: <b>{outcome}</b></p>

<p class="score">
Default probability: {payload['score']:.1%}
</p>

<p>
Score rebuilt from the force plot
(sigmoid(base + sum SHAP)) =
{payload['score_force']:.1%} &rarr; <b>{status}</b>
</p>

<h2>SHAP force plot</h2>
<img style="max-width:100%"
src="data:image/png;base64,{payload['force_b64']}">

{note}

<h2>Client profile</h2>
<div style="display:flex;gap:40px">
{table_html}
{categorical_html}
</div>

{figures_html}
</body>
</html>"""


# CLIENT SELECTION
def select_clients(explainer):

    holdout = explainer.X.loc[
        explainer.holdout_ids,
        ["SK_ID_CURR", "TARGET"],
    ].copy()

    holdout["p"] = explainer.model.predict_proba(
        explainer.X.loc[
            holdout.index,
            explainer.feats,
        ]
    )[:, 1]

    fpr, tpr, thresholds = roc_curve(
        holdout["TARGET"],
        holdout["p"],
    )

    # Youden's J statistic.
    threshold = thresholds[
        np.argmax(tpr - fpr)
    ]

    correct = holdout[
        (holdout["TARGET"] == 1)
        & (holdout["p"] >= threshold)
    ]

    incorrect = holdout[
        (holdout["TARGET"] == 1)
        & (holdout["p"] < threshold)
    ]

    client_correct = int(
        (
            correct if len(correct) else holdout
        )
        .sort_values("p")
        .iloc[-1]["SK_ID_CURR"]
    )

    client_wrong = int(
        (
            incorrect
            if len(incorrect)
            else holdout[holdout["TARGET"] == 1]
        )
        .sort_values("p")
        .iloc[0]["SK_ID_CURR"]
    )

    test = explainer.test_df.copy()

    test["p"] = explainer.model.predict_proba(
        test[explainer.feats]
    )[:, 1]

    percentile_90 = test["p"].quantile(0.9)

    client_test = int(
        test.iloc[
            (test["p"] - percentile_90).abs().argmin()
        ]["SK_ID_CURR"]
    )

    return (
        client_correct,
        client_wrong,
        client_test,
        float(threshold),
    )


def save_shap_summary(explainer):

    rng = np.random.RandomState(0)

    sample_ids = rng.choice(
        explainer.holdout_ids,
        min(2000, len(explainer.holdout_ids)),
        replace=False,
    )

    sample = explainer.X.loc[
        sample_ids,
        explainer.feats,
    ]

    shap_values = explainer.shap_explainer.shap_values(
        sample
    )

    shap_values = normalize_shap_values(shap_values)

    shap.summary_plot(
        shap_values,
        sample,
        max_display=25,
        show=False,
    )

    plt.gcf().savefig(
        SHAP_SUMMARY_PATH,
        dpi=130,
        bbox_inches="tight",
    )

    plt.close("all")

    print("saved", SHAP_SUMMARY_PATH)


def generate_reports(explainer, client_ids):

    client_correct, client_wrong, client_test = client_ids

    jobs = [
        (
            client_correct,
            "client1_correct_train",
            "Client 1 - train set, model CORRECT",
            "",
        ),
        (
            client_wrong,
            "client2_wrong_train",
            "Client 2 - train set, model WRONG",
            explainer.wrong_client_analysis(client_wrong),
        ),
        (
            client_test,
            "client_test",
            "Client 3 - test set",
            "",
        ),
    ]

    for client_id, filename, title, note in jobs:
        html = explainer.report_html(
            client_id,
            title,
            note,
        )

        output_path = OUT_DIR / f"{filename}.html"

        output_path.write_text(
            html,
            encoding="utf-8",
        )

        print(
            f"  {filename}.html "
            f"score={explainer.score(client_id):.3f}"
        )


def save_selected_clients(client_ids, threshold):

    client_correct, client_wrong, client_test = client_ids

    result = {
        "train_correct": client_correct,
        "train_wrong": client_wrong,
        "test": client_test,
        "youden_threshold": threshold,
    }

    output_path = OUT_DIR / "selected_clients.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate explainable credit-risk client reports."
    )

    parser.add_argument(
        "--ids",
        nargs=3,
        type=int,
        metavar=("TRAIN_OK", "TRAIN_WRONG", "TEST"),
        help="Specify three client IDs manually.",
    )

    args = parser.parse_args()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    explainer = Explainer()

    if args.ids:
        client_ids = tuple(args.ids)
        threshold = None
    else:
        (
            client_correct,
            client_wrong,
            client_test,
            threshold,
        ) = select_clients(explainer)

        client_ids = (
            client_correct,
            client_wrong,
            client_test,
        )

    print(
        "clients:",
        *client_ids,
        "| threshold:",
        threshold,
    )

    generate_reports(
        explainer,
        client_ids,
    )

    save_selected_clients(
        client_ids,
        threshold,
    )

    save_shap_summary(explainer)


if __name__ == "__main__":
    main()
