from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
MPLCONFIG_DIR = BASE_DIR / ".matplotlib_cache"
MPLCONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Topic_Rating_Differenz_Reviewebene_Enger_Filter import (
    DATA_DIR,
    GOOGLE_SOURCE_FILE,
    OUTPUT_DIR,
    TOPIC_COLS,
    YELP_TIGHT_SOURCE_FILE,
    load_tight_reference_counts,
    prepare_reviews,
)


OUTPUT_XLSX = OUTPUT_DIR / "topic_erwaehnungen_ratingklassen_enger_filter.xlsx"
OUTPUT_PNG = OUTPUT_DIR / "topic_erwaehnungen_ratingklassen_enger_filter.png"

RATING_CLASS_LABELS = [
    "1.0-1.9",
    "2.0-2.9",
    "3.0-3.9",
    "4.0-4.4",
    "4.5-5.0",
]

RED_PALETTE = [
    "#fff5f5", "#f9caca", "#f29a9a", "#e65f5f", "#c92a2a",
    "#a51111", "#7f0000", "#4d0000", "#111111",
]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 12
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"


def load_store_ratings():
    yelp = pd.read_excel(YELP_TIGHT_SOURCE_FILE)
    yelp = yelp.rename(columns={
        "business_id": "store_id",
        "review_stars": "rating",
    })
    yelp["source_dataset"] = "Yelp"

    google = pd.read_excel(GOOGLE_SOURCE_FILE, sheet_name="Alle_Daten")
    google["source_dataset"] = "Google"

    ratings = pd.concat(
        [
            yelp[["store_id", "rating", "business_name", "source_dataset"]],
            google[["store_id", "rating", "business_name", "source_dataset"]],
        ],
        ignore_index=True,
    )
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings[ratings["rating"].between(1, 5)].copy()

    return (
        ratings
        .groupby(["source_dataset", "store_id"], dropna=False)
        .agg(
            sternrating_filiale=("rating", "mean"),
            ratings_gesamt=("rating", "count"),
            filiale_name=("business_name", "first"),
        )
        .reset_index()
    )


def add_rating_class(df):
    df = df.copy()
    df["rating_klasse"] = pd.cut(
        df["sternrating_filiale"],
        bins=[1.0, 2.0, 3.0, 4.0, 4.5, 5.0],
        labels=RATING_CLASS_LABELS,
        include_lowest=True,
        right=False,
    )
    df.loc[df["sternrating_filiale"] == 5.0, "rating_klasse"] = RATING_CLASS_LABELS[-1]
    return df


def build_store_topic_table(reviews, store_ratings):
    for topic in TOPIC_COLS:
        reviews[f"{topic}_mentioned"] = pd.to_numeric(
            reviews[f"{topic}_mentioned"], errors="coerce"
        ).fillna(0)

    store_topics = (
        reviews
        .groupby(["source_dataset", "store_id"], dropna=False)
        .agg(
            reviews_mit_absa=("rating", "count"),
            **{f"{topic}_mentions": (f"{topic}_mentioned", "sum") for topic in TOPIC_COLS},
        )
        .reset_index()
    )

    store_table = store_ratings.merge(
        store_topics,
        on=["source_dataset", "store_id"],
        how="left",
    )
    store_table["reviews_mit_absa"] = store_table["reviews_mit_absa"].fillna(0).astype(int)

    for topic in TOPIC_COLS:
        store_table[f"{topic}_mentions"] = store_table[f"{topic}_mentions"].fillna(0)
        store_table[f"{topic}_mentions_pro_100_reviews"] = np.where(
            store_table["reviews_mit_absa"] > 0,
            store_table[f"{topic}_mentions"] / store_table["reviews_mit_absa"] * 100,
            0.0,
        )

    return add_rating_class(store_table)


def build_class_topic_table(store_table):
    grouped = (
        store_table
        .groupby("rating_klasse", observed=False)
        .agg(
            filialen=("store_id", "count"),
            ratings_gesamt=("ratings_gesamt", "sum"),
            reviews_mit_absa=("reviews_mit_absa", "sum"),
            **{f"{topic}_mentions": (f"{topic}_mentions", "sum") for topic in TOPIC_COLS},
        )
        .reindex(RATING_CLASS_LABELS)
        .fillna(0)
        .reset_index()
    )

    for topic in TOPIC_COLS:
        grouped[f"{topic}_mentions_pro_100_reviews"] = np.where(
            grouped["reviews_mit_absa"] > 0,
            grouped[f"{topic}_mentions"] / grouped["reviews_mit_absa"] * 100,
            0.0,
        )

    return grouped


def plot_class_topic_table(class_table):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(class_table["rating_klasse"]))
    width = 0.075

    for idx, topic in enumerate(TOPIC_COLS):
        offset = (idx - (len(TOPIC_COLS) - 1) / 2) * width
        ax.bar(
            x + offset,
            class_table[f"{topic}_mentions_pro_100_reviews"],
            width=width,
            label=topic,
            color=RED_PALETTE[idx % len(RED_PALETTE)],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_title("Sentiment-Topic-Erwähnungen nach Filial-Ratingklasse")
    ax.set_xlabel("Ratingklasse nach durchschnittlichem Filialrating")
    ax.set_ylabel("Erwähnungen pro 100 Reviews")
    ax.set_xticks(x)
    ax.set_xticklabels(class_table["rating_klasse"])
    ax.grid(False)
    legend = ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="black",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(OUTPUT_PNG, dpi=250)
    plt.close(fig)


def main():
    reference = load_tight_reference_counts()
    reviews = prepare_reviews()
    store_ratings = load_store_ratings()
    store_table = build_store_topic_table(reviews, store_ratings)
    class_table = build_class_topic_table(store_table)

    summary = pd.DataFrame([
        {"kennzahl": "ratings_referenzdatei", "wert": reference["ratings_total_reference"]},
        {"kennzahl": "filialen_referenzdatei", "wert": reference["restaurants_total_reference"]},
        {"kennzahl": "filialen_in_ratingklassen", "wert": len(store_table)},
        {"kennzahl": "reviews_mit_absa", "wert": len(reviews)},
        {"kennzahl": "google_reviews_mit_absa", "wert": int((reviews["source_dataset"] == "Google").sum())},
        {"kennzahl": "yelp_reviews_mit_absa", "wert": int((reviews["source_dataset"] == "Yelp").sum())},
    ])

    with pd.ExcelWriter(OUTPUT_XLSX) as writer:
        class_table.to_excel(writer, sheet_name="ratingklassen", index=False)
        store_table.to_excel(writer, sheet_name="filialen", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

    plot_class_topic_table(class_table)

    print("FERTIG")
    print(f"Excel: {OUTPUT_XLSX}")
    print(f"Plot: {OUTPUT_PNG}")
    print(f"Filialen: {len(store_table):,}")
    print(f"Reviews mit ABSA: {len(reviews):,}")


if __name__ == "__main__":
    main()
