from pathlib import Path
import os
import re

BASE_DIR = Path(__file__).resolve().parent
MPLCONFIG_DIR = BASE_DIR / ".matplotlib_cache"
MPLCONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# KONFIGURATION
# =========================================================

PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
SERVQUAL_OUTPUT_DIR = BASE_DIR / "servqual_outputs"
OUTPUT_DIR = BASE_DIR / "topic_class_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

ABSA_WASHINGTON_FILE = DATA_DIR / "absa_review_level_washington2.xlsx"
ABSA_STRENG_FILE = DATA_DIR / "absa_review_level_streng2.xlsx"
GOOGLE_SOURCE_FILE = DATA_DIR / "ChickFilA_Bereinigt.xlsx"
YELP_SOURCE_FILE = SERVQUAL_OUTPUT_DIR / "chick_fil_a_yelp_filtered.csv"

STORE_TOPIC_TABLE_FILE = OUTPUT_DIR / "topic_erwaehnungen_filialen.xlsx"
CLASS_TOPIC_TABLE_FILE = OUTPUT_DIR / "topic_erwaehnungen_ratingklassen.xlsx"
TOPIC_RATING_EFFECTS_FILE = OUTPUT_DIR / "topic_rating_effects.xlsx"
TOPIC_RATING_EFFECTS_PLOT_FILE = OUTPUT_DIR / "topic_rating_effects.png"

RATING_CLASS_LABELS = [
    "1.0-1.9",
    "2.0-2.9",
    "3.0-3.9",
    "4.0-4.4",
    "4.5-5.0",
]

TOPIC_COLS = [
    "FOOD", "DRINKS", "SERVICE", "SPEED", "HYGIENE", "VALUE",
    "AMBIENCE", "PARKING", "ACCESSIBILITY",
]

RED_PALETTE = [
    "#fff5f5", "#f9caca", "#f29a9a", "#e65f5f", "#c92a2a",
    "#a51111", "#7f0000", "#4d0000", "#2b0000", "#111111",
]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 12
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"


# =========================================================
# HILFSFUNKTIONEN
# =========================================================

def normalize_user_id(value):
    if pd.isna(value):
        return None
    if isinstance(value, float):
        return format(value, ".0f")
    value = str(value).strip()
    if value.endswith(".0") and value[:-2].isdigit():
        return value[:-2]
    return value


def user_prefix(value, n=6):
    normalized = normalize_user_id(value)
    return normalized[:n] if normalized else None


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


def style_plot(ax):
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")
    ax.grid(False)
    ax.tick_params(colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")
    ax.title.set_fontsize(15)
    ax.xaxis.label.set_fontsize(13)
    ax.yaxis.label.set_fontsize(13)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_color("black")


# =========================================================
# REVIEW-LEVEL DATEN AUS ABSA + ORIGINALRATINGS MERGEN
# =========================================================

def load_yelp_review_level():
    absa = pd.read_excel(ABSA_STRENG_FILE)
    source = pd.read_csv(YELP_SOURCE_FILE, encoding="utf-8-sig")

    absa["time_key"] = pd.to_datetime(absa["time"], errors="coerce")
    source["time_key"] = pd.to_datetime(source["time"], errors="coerce")

    merged = absa.merge(
        source,
        on=["user_id", "store_id", "time_key"],
        how="left",
        suffixes=("", "_source"),
    )
    merged["source_dataset"] = "yelp"
    merged["rating"] = pd.to_numeric(merged["rating"], errors="coerce")
    return merged


def load_google_review_level():
    absa = pd.read_excel(ABSA_WASHINGTON_FILE)
    source = pd.read_excel(GOOGLE_SOURCE_FILE, sheet_name="Alle_Daten")

    absa["user_prefix6"] = absa["user_id"].map(user_prefix)
    source["user_prefix6"] = source["user_id"].map(user_prefix)
    absa["time_key"] = pd.to_datetime(absa["time"], errors="coerce").dt.floor("s")
    source["time_key"] = pd.to_datetime(source["time"], errors="coerce").dt.floor("s")

    merged = absa.merge(
        source,
        on=["user_prefix6", "store_id", "time_key"],
        how="left",
        suffixes=("_absa", ""),
    )
    merged["source_dataset"] = "google_washington"
    merged["rating"] = pd.to_numeric(merged["rating"], errors="coerce")
    return merged


def prepare_reviews():
    yelp = load_yelp_review_level()
    google = load_google_review_level()
    common_cols = sorted(set(yelp.columns) | set(google.columns))
    reviews = pd.concat(
        [yelp.reindex(columns=common_cols), google.reindex(columns=common_cols)],
        ignore_index=True,
    )

    reviews = reviews[reviews["rating"].between(1, 5)].copy()
    for topic in TOPIC_COLS:
        reviews[topic] = pd.to_numeric(reviews[topic], errors="coerce")
        reviews[f"{topic}_mentioned"] = reviews[topic].notna().astype(int)

    return reviews


# =========================================================
# AGGREGATION
# =========================================================

def build_store_topic_table(reviews):
    agg_dict = {
        "rating": ["mean", "count"],
    }
    if "business_name" in reviews.columns:
        agg_dict["business_name"] = "first"
    if "address" in reviews.columns:
        agg_dict["address"] = "first"
    if "source_dataset" in reviews.columns:
        agg_dict["source_dataset"] = "first"

    for topic in TOPIC_COLS:
        agg_dict[f"{topic}_mentioned"] = "sum"
        agg_dict[topic] = "mean"

    store_table = reviews.groupby("store_id", dropna=False).agg(agg_dict)
    store_table.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in store_table.columns
    ]
    store_table = store_table.reset_index()

    store_table = store_table.rename(columns={
        "rating_mean": "sternrating_filiale",
        "rating_count": "anzahl_reviews",
        "business_name_first": "filiale_name",
        "address_first": "adresse",
        "source_dataset_first": "source_dataset",
    })

    for topic in TOPIC_COLS:
        mention_col = f"{topic}_mentioned_sum"
        store_table = store_table.rename(columns={mention_col: f"{topic}_mentions"})
        store_table[f"{topic}_mentions_pro_100_reviews"] = np.where(
            store_table["anzahl_reviews"] > 0,
            store_table[f"{topic}_mentions"] / store_table["anzahl_reviews"] * 100,
            0.0,
        )
        store_table = store_table.rename(columns={f"{topic}_mean": f"{topic}_sentiment_mean"})

    store_table = add_rating_class(store_table)
    return store_table.sort_values(["sternrating_filiale", "anzahl_reviews"], ascending=[False, False])


def build_class_topic_table(store_table):
    mention_cols = [f"{topic}_mentions" for topic in TOPIC_COLS]
    grouped = (
        store_table
        .groupby("rating_klasse", observed=False)[mention_cols + ["anzahl_reviews"]]
        .sum()
        .reindex(RATING_CLASS_LABELS)
        .fillna(0)
    )

    class_table = pd.DataFrame(index=grouped.index)
    class_table["anzahl_reviews"] = grouped["anzahl_reviews"]
    for topic in TOPIC_COLS:
        class_table[f"{topic}_mentions"] = grouped[f"{topic}_mentions"]
        class_table[f"{topic}_mentions_pro_100_reviews"] = np.where(
            grouped["anzahl_reviews"] > 0,
            grouped[f"{topic}_mentions"] / grouped["anzahl_reviews"] * 100,
            0.0,
        )

    return class_table.reset_index()


def build_topic_rating_effects(reviews):
    rows = []
    for topic in TOPIC_COLS:
        mention_col = f"{topic}_mentioned"
        sentiment = pd.to_numeric(reviews[topic], errors="coerce")
        rating = pd.to_numeric(reviews["rating"], errors="coerce")
        mentioned = reviews[mention_col].astype(float)

        rating_when_mentioned = rating[mentioned == 1].mean()
        rating_when_not_mentioned = rating[mentioned == 0].mean()
        mention_rating_diff = rating_when_mentioned - rating_when_not_mentioned
        mention_corr = rating.corr(mentioned)
        sentiment_corr = rating.corr(sentiment)

        valid = sentiment.notna() & rating.notna()
        if valid.sum() >= 3 and sentiment[valid].nunique() > 1:
            slope, intercept = np.polyfit(sentiment[valid], rating[valid], 1)
        else:
            slope, intercept = np.nan, np.nan

        rows.append({
            "topic": topic,
            "reviews_total": int(rating.notna().sum()),
            "reviews_with_topic": int((mentioned == 1).sum()),
            "topic_mention_share": mentioned.mean(),
            "rating_when_topic_mentioned": rating_when_mentioned,
            "rating_when_topic_not_mentioned": rating_when_not_mentioned,
            "mention_rating_difference": mention_rating_diff,
            "mention_rating_correlation": mention_corr,
            "topic_sentiment_rating_correlation": sentiment_corr,
            "topic_sentiment_rating_slope": slope,
            "topic_sentiment_rating_intercept": intercept,
        })

    effects = pd.DataFrame(rows)
    return effects.sort_values("mention_rating_difference", ascending=False)


def plot_topic_rating_effects(effects):
    values = effects.sort_values("mention_rating_difference", ascending=True)
    colors = np.where(values["mention_rating_difference"] >= 0, "#b71c1c", "#222222")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(values))
    ax.barh(
        y,
        values["mention_rating_difference"],
        color=colors,
        edgecolor="black",
        linewidth=0.7,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(values["topic"])
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Rating-Differenz bei Topic-Erwähnung")
    ax.set_xlabel("Durchschnittsrating: erwähnt minus nicht erwähnt")
    ax.set_ylabel("Sentiment-Topic")
    style_plot(ax)
    fig.tight_layout()
    fig.savefig(TOPIC_RATING_EFFECTS_PLOT_FILE, dpi=250)
    plt.close(fig)


# =========================================================
# PLOTS
# =========================================================

def plot_single_rating_classes(class_table):
    for _, row in class_table.iterrows():
        class_label = row["rating_klasse"]
        values = pd.Series({
            topic: row[f"{topic}_mentions_pro_100_reviews"]
            for topic in TOPIC_COLS
        }).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 5.5))
        x = np.arange(len(values.index))
        ax.bar(x, values.values, color="#b71c1c", edgecolor="black", linewidth=0.8)
        ax.set_title(f"Topic-Erwähnungen: Ratingklasse {class_label}")
        ax.set_xlabel("Sentiment-Topic")
        ax.set_ylabel("Erwähnungen pro 100 Reviews")
        ax.set_xticks(x)
        ax.set_xticklabels(values.index, rotation=30, ha="right")
        style_plot(ax)
        fig.tight_layout()
        safe_label = str(class_label).replace(".", "_").replace("-", "_")
        fig.savefig(OUTPUT_DIR / f"topic_ratingklasse_{safe_label}.png", dpi=250)
        plt.close(fig)


def plot_aggregated_rating_classes(class_table):
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
    style_plot(ax)
    legend = ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="black",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
    )
    for text in legend.get_texts():
        text.set_color("black")
        text.set_fontweight("bold")
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(OUTPUT_DIR / "topic_ratingklassen_aggregiert.png", dpi=250)
    plt.close(fig)


def plot_heatmap(class_table):
    value_cols = [f"{topic}_mentions_pro_100_reviews" for topic in TOPIC_COLS]
    values = class_table[value_cols].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    im = ax.imshow(values, cmap="Reds", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Erwähnungen pro 100 Reviews", fontweight="bold")
    ax.set_yticks(np.arange(len(class_table["rating_klasse"])))
    ax.set_yticklabels(class_table["rating_klasse"])
    ax.set_xticks(np.arange(len(TOPIC_COLS)))
    ax.set_xticklabels(TOPIC_COLS, rotation=30, ha="right")
    ax.set_title("Topic-Erwähnungen je Ratingklasse")
    style_plot(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "topic_ratingklassen_heatmap.png", dpi=250)
    plt.close(fig)


def main():
    reviews = prepare_reviews()
    store_table = build_store_topic_table(reviews)
    class_table = build_class_topic_table(store_table)
    effects = build_topic_rating_effects(reviews)

    store_table.to_excel(STORE_TOPIC_TABLE_FILE, index=False)
    class_table.to_excel(CLASS_TOPIC_TABLE_FILE, index=False)
    effects.to_excel(TOPIC_RATING_EFFECTS_FILE, index=False)

    plot_single_rating_classes(class_table)
    plot_aggregated_rating_classes(class_table)
    plot_heatmap(class_table)
    plot_topic_rating_effects(effects)

    print("\nFERTIG")
    print(f"Review-Level-Zeilen: {len(reviews):,}")
    print(f"Filialen: {len(store_table):,}")
    print(f"Filialtabelle: {STORE_TOPIC_TABLE_FILE}")
    print(f"Ratingklassen-Tabelle: {CLASS_TOPIC_TABLE_FILE}")
    print(f"Topic-Rating-Effekte: {TOPIC_RATING_EFFECTS_FILE}")
    print(f"Plots: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
