import json
import textwrap
from pathlib import Path

ROOT = Path(".").resolve()

def to_lines(text):
    text = textwrap.dedent(text).strip("\n")
    if not text:
        return []
    return [line + "\n" for line in text.splitlines()]

def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": to_lines(text),
    }

def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_lines(text),
    }

cells = [
    md("""
    # My Vivino
    ## Content-Based Wine Recommendation MVP

    **Authors:** Leila Cheikhe / Hazem Barman  
    **Track:** Qwasar Season 02 Data Science

    ### Approved project
    Build a content-based wine recommendation MVP for My Vivino.

    ### Business goal
    Replace the current rules-based recommendation system with a data-driven MVP that can:
    - recommend similar wines for cold-start inventory
    - identify high-value wines for promotion
    - produce measurable outputs for product and leadership
    - stay simple enough for DevOps to deploy
    """),

    md("""
    ## Scientific framing

    ### Problem
    My Vivino has a large catalog and millions of users, but its recommendation logic is still mostly rules-based.

    ### Hypotheses
    - **H1:** Wine descriptions and metadata contain enough signal to predict wine variety better than a naive baseline.
    - **H2:** A value score combining quality and price can identify attractive wines for merchandising.
    - **H3:** A content-based nearest-neighbor model can generate useful recommendations without user history.

    ### Success criteria
    - classification accuracy >= 0.60
    - top-3 accuracy >= 0.85
    - at least 5,000 wines available for cold-start recommendation
    - a clear notebook story for product, CEO, and DevOps
    """),

    md("""
    ## Data source

    This notebook uses the public Wine Reviews dataset.

    - Kaggle source: https://www.kaggle.com/datasets/zynicide/wine-reviews
    - Expected local file: `data/winemag-data-130k-v2.csv`
    - If the file is missing, the notebook tries public raw CSV mirrors automatically
    """),

    code("""
    from pathlib import Path
    import math
    import urllib.request

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import NearestNeighbors
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.svm import LinearSVC

    sns.set_style("whitegrid")
    pd.set_option("display.max_columns", 50)

    DATA_DIR = Path("data")
    DATA_DIR.mkdir(exist_ok=True)
    DATA_PATH = DATA_DIR / "winemag-data-130k-v2.csv"

    DATA_URLS = [
        "https://raw.githubusercontent.com/rajacsp/public-dataset/master/zynicide-wine-reviews/winemag-data-130k-v2.csv",
        "https://gist.githubusercontent.com/clairehq/79acab35be50eaf1c383948ed3fd1129/raw/winemag-data-130k-v2.csv",
    ]
    """),

    code("""
    def download_if_needed(path, urls):
        if path.exists():
            print("Using local dataset:", path)
            return path

        last_error = None
        for url in urls:
            try:
                print("Trying to download dataset from:", url)
                urllib.request.urlretrieve(url, str(path))
                print("Downloaded dataset to:", path)
                return path
            except Exception as exc:
                last_error = exc

        raise FileNotFoundError(
            "Could not download the dataset automatically. "
            "Please place winemag-data-130k-v2.csv inside data/."
        ) from last_error

    dataset_path = download_if_needed(DATA_PATH, DATA_URLS)
    raw_df = pd.read_csv(dataset_path)
    raw_df.head()
    """),

    code("""
    raw_df.shape
    """),

    md("""
    ## Data cleaning

    Keep the fields that matter for recommendation and ranking:
    - `description`
    - `points`
    - `price`
    - `country`
    - `province`
    - `title`
    - `variety`
    - `winery`
    """),

    code("""
    df = raw_df.copy()

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    keep_columns = [
        "country",
        "description",
        "points",
        "price",
        "province",
        "title",
        "variety",
        "winery",
    ]
    df = df[keep_columns].copy()

    df = df.dropna(subset=["description", "points", "title", "variety", "winery"])
    df = df.drop_duplicates(subset=["title", "description", "winery"])

    df["country"] = df["country"].fillna("Unknown")
    df["province"] = df["province"].fillna("Unknown")

    country_medians = df.groupby("country")["price"].transform("median")
    global_price_median = df["price"].median()
    df["price"] = df["price"].fillna(country_medians).fillna(global_price_median)

    df["description_length"] = df["description"].str.split().str.len()
    df["log_price"] = np.log1p(df["price"])

    df["points_z"] = (df["points"] - df["points"].mean()) / df["points"].std(ddof=0)
    df["price_z"] = (df["log_price"] - df["log_price"].mean()) / df["log_price"].std(ddof=0)
    df["value_score"] = df["points_z"] - df["price_z"]

    df["price_band"] = pd.cut(
        df["price"],
        bins=[0, 15, 30, 60, 100, np.inf],
        labels=["budget", "entry", "mid", "premium", "luxury"],
        include_lowest=True,
    )

    print(df.shape)
    df.head()
    """),

    code("""
    summary = pd.DataFrame({
        "rows": [len(df)],
        "countries": [df["country"].nunique()],
        "varieties": [df["variety"].nunique()],
        "wineries": [df["winery"].nunique()],
        "median_price": [round(df["price"].median(), 2)],
        "median_points": [round(df["points"].median(), 2)],
    })
    summary
    """),

    md("""
    ## Exploratory data analysis
    """),

    code("""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df["points"], bins=25, kde=True, ax=axes[0], color="#8c2d19")
    axes[0].set_title("Distribution of wine scores")
    axes[0].set_xlabel("Points")

    sns.histplot(df.loc[df["price"] <= 150, "price"], bins=30, kde=True, ax=axes[1], color="#3f7cac")
    axes[1].set_title("Price distribution (<= $150)")
    axes[1].set_xlabel("Price")

    plt.tight_layout()
    plt.show()
    """),

    code("""
    top_countries = df["country"].value_counts().head(10).sort_values()
    top_varieties = df["variety"].value_counts().head(12).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    top_countries.plot(kind="barh", ax=axes[0], color="#7f5539")
    axes[0].set_title("Top countries by review count")
    axes[0].set_xlabel("Reviews")

    top_varieties.plot(kind="barh", ax=axes[1], color="#bc4749")
    axes[1].set_title("Top wine varieties")
    axes[1].set_xlabel("Reviews")

    plt.tight_layout()
    plt.show()
    """),

    code("""
    sample_df = df.sample(min(len(df), 5000), random_state=42)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=sample_df,
        x="price",
        y="points",
        hue="price_band",
        alpha=0.65,
        s=40,
    )
    plt.xlim(0, 150)
    plt.title("Price vs score sample")
    plt.tight_layout()
    plt.show()
    """),

    md("""
    ## Merchandising angle: value wines

    We define:

    `value_score = standardized(points) - standardized(log(price))`

    High values indicate wines that score well without being too expensive.
    """),

    code("""
    high_value_wines = (
        df.loc[(df["price"] <= 60) & (df["points"] >= 88)]
        .sort_values(["value_score", "points"], ascending=[False, False])
        .drop_duplicates(subset=["title"])
        .loc[:, ["title", "country", "variety", "price", "points", "value_score"]]
        .head(15)
    )
    high_value_wines
    """),

    md("""
    ## Machine learning experiment

    Predict wine variety using:
    - `description`
    - `country`
    - `province`
    - `price`
    - `points`
    - `description_length`
    """),

    code("""
    variety_counts = df["variety"].value_counts()
    target_varieties = variety_counts[variety_counts >= 500].head(12).index.tolist()

    model_df = df[df["variety"].isin(target_varieties)].copy()

    features = ["description", "country", "province", "price", "points", "description_length"]
    target = "variety"

    X_train, X_test, y_train, y_test = train_test_split(
        model_df[features],
        model_df[target],
        test_size=0.2,
        random_state=42,
        stratify=model_df[target],
    )

    baseline_accuracy = y_train.value_counts(normalize=True).max()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=8000,
                    ngram_range=(1, 2),
                    min_df=5,
                ),
                "description",
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["country", "province"]),
            ("num", StandardScaler(with_mean=False), ["price", "points", "description_length"]),
        ]
    )

    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LinearSVC(random_state=42)),
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    decision_scores = clf.decision_function(X_test)
    class_names = clf.named_steps["classifier"].classes_

    if len(class_names) < 3 or len(getattr(decision_scores, "shape", [])) == 1:
        top3_accuracy = float("nan")
    else:
        top_indices = np.argsort(decision_scores, axis=1)[:, -3:]
        top_labels = class_names[top_indices]
        top3_accuracy = np.mean([
            true_label in predicted_labels
            for true_label, predicted_labels in zip(y_test.values, top_labels)
        ])

    metrics_table = pd.DataFrame({
        "metric": ["majority_baseline_accuracy", "model_accuracy", "macro_f1", "top_3_accuracy"],
        "value": [baseline_accuracy, accuracy, macro_f1, top3_accuracy],
    })
    metrics_table
    """),

    code("""
    report_df = (
        pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0))
        .T.sort_values("f1-score", ascending=False)
    )
    report_df.head(15)
    """),

    code("""
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, cmap="YlOrBr", xticklabels=labels, yticklabels=labels)
    plt.title("Variety classification confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    """),

    md("""
    ## Recommendation prototype

    Build a content-based recommender for cold-start inventory:
    - keep wines with `price <= 80` and `points >= 88`
    - vectorize text plus metadata
    - retrieve nearest neighbors with cosine similarity
    """),

    code("""
    reco_df = (
        df.loc[(df["price"] <= 80) & (df["points"] >= 88)]
        .sort_values(["value_score", "points"], ascending=[False, False])
        .drop_duplicates(subset=["title"])
        .head(8000)
        .reset_index(drop=True)
    )

    if len(reco_df) < 2:
        raise ValueError("Not enough wines available for recommendation demo.")

    reco_text = (
        reco_df["description"].fillna("") + " " +
        reco_df["variety"].fillna("") + " " +
        reco_df["country"].fillna("") + " " +
        reco_df["province"].fillna("")
    )

    reco_vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=6000,
        ngram_range=(1, 2),
        min_df=3,
    )
    reco_matrix = reco_vectorizer.fit_transform(reco_text)

    recommender = NearestNeighbors(metric="cosine", n_neighbors=min(6, len(reco_df)))
    recommender.fit(reco_matrix)

    def recommend_from_index(seed_index, n_results=5):
        total = min(n_results + 1, len(reco_df))
        distances, indices = recommender.kneighbors(reco_matrix[seed_index], n_neighbors=total)
        chosen = reco_df.loc[[seed_index], ["title", "country", "variety", "price", "points", "value_score"]]
        recommendations = reco_df.loc[
            indices[0][1:], ["title", "country", "variety", "price", "points", "value_score"]
        ].copy()
        recommendations["similarity"] = (1 - distances[0][1:]).round(3)
        return chosen, recommendations.sort_values("similarity", ascending=False)

    seed_index = int(reco_df["value_score"].idxmax())
    chosen_wine, recommendations = recommend_from_index(seed_index, 5)
    chosen_wine, recommendations
    """),

    md("""
    ## Business impact simulation

    These are scenario estimates, not observed production results.
    They help leadership reason about possible ROI.
    """),

    code("""
    scenario_df = pd.DataFrame([
        {"scenario": "Conservative", "exposed_share": 0.20, "baseline_conversion": 0.03, "relative_lift": 0.02, "avg_order_value": 28},
        {"scenario": "Base", "exposed_share": 0.30, "baseline_conversion": 0.04, "relative_lift": 0.04, "avg_order_value": 30},
        {"scenario": "Upside", "exposed_share": 0.40, "baseline_conversion": 0.05, "relative_lift": 0.06, "avg_order_value": 32},
    ])

    monthly_users = 27000000

    scenario_df["incremental_monthly_orders"] = (
        monthly_users
        * scenario_df["exposed_share"]
        * scenario_df["baseline_conversion"]
        * scenario_df["relative_lift"]
    ).round(0)

    scenario_df["incremental_monthly_revenue_usd"] = (
        scenario_df["incremental_monthly_orders"] * scenario_df["avg_order_value"]
    ).round(2)

    scenario_df["incremental_annual_revenue_usd"] = (
        scenario_df["incremental_monthly_revenue_usd"] * 12
    ).round(2)

    scenario_df
    """),

    code("""
    plt.figure(figsize=(9, 5))
    sns.barplot(data=scenario_df, x="scenario", y="incremental_annual_revenue_usd")
    plt.title("Illustrative annual revenue impact")
    plt.ylabel("Incremental annual revenue (USD)")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()
    """),

    code("""
    success_table = pd.DataFrame([
        {
            "criterion": "Variety model accuracy >= 0.60",
            "observed": round(float(accuracy), 3),
            "status": "PASS" if accuracy >= 0.60 else "NEEDS WORK",
        },
        {
            "criterion": "Top-3 accuracy >= 0.85",
            "observed": round(float(top3_accuracy), 3) if not np.isnan(top3_accuracy) else np.nan,
            "status": "PASS" if (not np.isnan(top3_accuracy) and top3_accuracy >= 0.85) else "NEEDS WORK",
        },
        {
            "criterion": "Cold-start catalog >= 5000 wines",
            "observed": int(len(reco_df)),
            "status": "PASS" if len(reco_df) >= 5000 else "NEEDS WORK",
        },
    ])
    success_table
    """),

    md("""
    ## Production plan for DevOps

    A realistic MVP release:
    - weekly offline retraining job
    - saved TF-IDF vectorizer, classifier, and nearest-neighbor index
    - lightweight API endpoint such as `/recommend?wine_id=...`
    - monitoring for coverage, latency, CTR, conversion lift, and drift

    ## Conclusion
    This experiment shows that My Vivino can move from static rules to a measurable recommendation MVP.

    ## Next steps
    - integrate user logs for personalization
    - run A/B tests against the legacy rules engine
    - add seasonality, food pairing, and inventory constraints
    - move toward conversion-optimized ranking
    """),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.6",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

README = """
# My Vivino

Data science MVP for a content-based wine recommendation system.

## Project idea
The approved project is a content-based wine recommendation engine for My Vivino.

It improves the legacy rules-based flow with:
- interpretable machine learning
- cold-start recommendations
- high-value wine discovery
- a production-friendly MVP path

## Main deliverable
- `my_vivino.ipynb`

## Supporting files
- `presentation_outline.md`
- `blog_post.md`
- `requirements.txt`
"""

PRESENTATION = """
# Presentation Outline

## Slide 1 - Why change the recommender
- The current system is rules-based
- New wines have a cold-start problem
- Leadership wants measurable improvement

## Slide 2 - Approved project
- Build a content-based recommendation MVP
- Use descriptions, price, geography, and quality signals
- Keep the solution explainable and production-friendly

## Slide 3 - Hypotheses
- Text and metadata can predict variety
- Value score can highlight affordable quality wines
- Similarity search can power cold-start recommendation

## Slide 4 - Data work
- Collect and clean the Wine Reviews dataset
- Handle missing prices and duplicate reviews
- Engineer value-oriented features

## Slide 5 - Exploration
- Scores are concentrated in the upper 80s and low 90s
- Prices are skewed
- A few countries and varieties dominate the catalog

## Slide 6 - ML results
- Compare model accuracy to a naive baseline
- Show macro F1 and top-3 accuracy
- Demonstrate that product text contains strong signal

## Slide 7 - Recommendation demo
- Pick one featured wine
- Return five similar wines
- Explain why this helps cold-start inventory

## Slide 8 - Business impact and production
- Show scenario-based revenue estimates
- Propose weekly retraining and API serving
- Recommend A/B testing against the old system
"""

BLOG = """
# My Vivino: From Rules-Based Recommendations to a Data Science MVP

## Introduction
My Vivino has a large wine catalog and millions of users, but its recommendation logic is still mostly rules-based.
That makes the system hard to improve, hard to measure, and weak on new inventory.

## Question
Can we build a recommendation MVP that is:
- explainable
- useful for cold-start recommendations
- valuable for product leadership
- realistic for DevOps to deploy

## Hypotheses
1. Wine descriptions and metadata contain enough signal to predict wine variety.
2. A value score can identify strong wines for merchandising.
3. Nearest-neighbor similarity can generate useful recommendations without user history.

## Data
This project uses the public Wine Reviews dataset with descriptions, prices, scores, countries, provinces, wineries, and varieties.

## Method
1. Collect data
2. Clean and validate it
3. Explore the main patterns
4. Train a supervised model
5. Build a recommendation prototype
6. Simulate business impact

## Business takeaway
This project shows that My Vivino can replace static recommendation rules with a measurable and explainable MVP.
"""

REQUIREMENTS = """
jupyter==1.0.0
numpy==1.19.5
pandas==1.1.5
matplotlib==3.3.4
seaborn==0.11.2
scikit-learn==0.24.2
"""

GITIGNORE = """
.venv/
__pycache__/
.ipynb_checkpoints/
data/*.csv
!data/.gitkeep
"""

def write_text(path, content):
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")

(ROOT / "data").mkdir(exist_ok=True)
write_text(ROOT / "data" / ".gitkeep", "")
write_text(ROOT / "README.md", README)
write_text(ROOT / "presentation_outline.md", PRESENTATION)
write_text(ROOT / "blog_post.md", BLOG)
write_text(ROOT / "requirements.txt", REQUIREMENTS)
write_text(ROOT / ".gitignore", GITIGNORE)
(ROOT / "my_vivino.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")

print("Generated project files.")
