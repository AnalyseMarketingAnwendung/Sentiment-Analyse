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


PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = BASE_DIR / "topic_class_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TIGHT_RATINGS_FILE = Path(
    "/Users/clara/Documents/New project/outputs/chickfila_ratings/chickfila_ratings_yelp_google.xlsx"
)
YELP_TIGHT_SOURCE_FILE = DATA_DIR / "chickfila_filtered.xlsx"
GOOGLE_SOURCE_FILE = DATA_DIR / "ChickFilA_Bereinigt.xlsx"
ABSA_YELP_FILE = DATA_DIR / "absa_review_level_streng2.xlsx"
ABSA_GOOGLE_FILE = DATA_DIR / "absa_review_level_washington2.xlsx"

OUTPUT_XLSX = OUTPUT_DIR / "topic_rating_effects_reviewebene_enger_filter.xlsx"
OUTPUT_PNG = OUTPUT_DIR / "topic_rating_effects_reviewebene_enger_filter.png"

TOPIC_COLS = [
    "FOOD", "DRINKS", "SERVICE", "SPEED", "HYGIENE", "VALUE",
    "AMBIENCE", "PARKING", "ACCESSIBILITY",
]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 12
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"


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


def load_tight_reference_counts():
    ratings = pd.read_excel(TIGHT_RATINGS_FILE, sheet_name="Ratings")
    return {
        "ratings_total_reference": len(ratings),
        "restaurants_total_reference": ratings["Restaurant"].nunique(dropna=True),
        "google_ratings_reference": int((ratings["Quelle"] == "Google").sum()),
        "yelp_ratings_reference": int((ratings["Quelle"] == "Yelp").sum()),
        "google_restaurants_reference": int(
            ratings.loc[ratings["Quelle"] == "Google", "Restaurant"].nunique()
        ),
        "yelp_restaurants_reference": int(
            ratings.loc[ratings["Quelle"] == "Yelp", "Restaurant"].nunique()
        ),
    }


def load_yelp_reviews():
    absa = pd.read_excel(ABSA_YELP_FILE)
    source = pd.read_excel(YELP_TIGHT_SOURCE_FILE)

    absa["time_key"] = pd.to_datetime(absa["time"], errors="coerce")
    source["time_key"] = pd.to_datetime(source["review_date"], errors="coerce")
    source = source.rename(columns={
        "business_id": "store_id",
        "review_stars": "rating",
    })

    merged = absa.merge(
        source[["user_id", "store_id", "time_key", "rating", "business_name"]],
        on=["user_id", "store_id", "time_key"],
        how="inner",
    )
    merged["source_dataset"] = "Yelp"
    return merged


def load_google_reviews():
    absa = pd.read_excel(ABSA_GOOGLE_FILE)
    source = pd.read_excel(GOOGLE_SOURCE_FILE, sheet_name="Alle_Daten")

    absa["user_prefix6"] = absa["user_id"].map(user_prefix)
    source["user_prefix6"] = source["user_id"].map(user_prefix)
    absa["time_key"] = pd.to_datetime(absa["time"], errors="coerce").dt.floor("s")
    source["time_key"] = pd.to_datetime(source["time"], errors="coerce").dt.floor("s")

    key_cols = ["user_prefix6", "store_id", "time_key"]
    absa = (
        absa
        .groupby(key_cols, dropna=False, as_index=False)
        .agg({**{topic: "mean" for topic in TOPIC_COLS}, "user_id": "first", "time": "first"})
    )

    merged = absa.merge(
        source[key_cols + ["rating", "business_name"]],
        on=key_cols,
        how="inner",
    )
    merged["source_dataset"] = "Google"
    return merged


def prepare_reviews():
    yelp = load_yelp_reviews()
    google = load_google_reviews()
    reviews = pd.concat([yelp, google], ignore_index=True, sort=False)
    reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")
    reviews = reviews[reviews["rating"].between(1, 5)].copy()

    for topic in TOPIC_COLS:
        reviews[topic] = pd.to_numeric(reviews[topic], errors="coerce")
        reviews[f"{topic}_mentioned"] = reviews[topic].notna().astype(int)

    return reviews


def build_topic_rating_effects(reviews):
    rows = []
    for topic in TOPIC_COLS:
        mentioned = reviews[f"{topic}_mentioned"].astype(float)
        rating = reviews["rating"]
        sentiment = reviews[topic]

        rows.append({
            "topic": topic,
            "reviews_total": int(rating.notna().sum()),
            "reviews_with_topic": int((mentioned == 1).sum()),
            "topic_mention_share": mentioned.mean(),
            "rating_when_topic_mentioned": rating[mentioned == 1].mean(),
            "rating_when_topic_not_mentioned": rating[mentioned == 0].mean(),
            "mention_rating_difference": rating[mentioned == 1].mean() - rating[mentioned == 0].mean(),
            "topic_sentiment_rating_correlation": rating.corr(sentiment),
        })

    return pd.DataFrame(rows).sort_values("mention_rating_difference", ascending=False)


def style_plot(ax):
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")
    ax.grid(False)
    ax.tick_params(colors="black")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_color("black")


def plot_topic_rating_effects(effects):
    values = effects.sort_values("mention_rating_difference", ascending=True)
    colors = np.where(values["mention_rating_difference"] >= 0, "#b71c1c", "#222222")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(values))
    ax.barh(y, values["mention_rating_difference"], color=colors, edgecolor="black", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(values["topic"])
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Rating-Differenz bei Topic-Erwähnung")
    ax.set_xlabel("Durchschnittsrating: erwähnt minus nicht erwähnt")
    ax.set_ylabel("Sentiment-Topic")
    style_plot(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=250)
    plt.close(fig)


def main():
    reference = load_tight_reference_counts()
    reviews = prepare_reviews()
    effects = build_topic_rating_effects(reviews)

    source_counts = reviews["source_dataset"].value_counts().to_dict()
    summary = pd.DataFrame([
        {"kennzahl": "ratings_referenzdatei", "wert": reference["ratings_total_reference"]},
        {"kennzahl": "filialen_referenzdatei", "wert": reference["restaurants_total_reference"]},
        {"kennzahl": "google_ratings_referenzdatei", "wert": reference["google_ratings_reference"]},
        {"kennzahl": "yelp_ratings_referenzdatei", "wert": reference["yelp_ratings_reference"]},
        {"kennzahl": "google_filialen_referenzdatei", "wert": reference["google_restaurants_reference"]},
        {"kennzahl": "yelp_filialen_referenzdatei", "wert": reference["yelp_restaurants_reference"]},
        {"kennzahl": "reviews_fuer_topic_grafik", "wert": len(reviews)},
        {"kennzahl": "google_reviews_fuer_topic_grafik", "wert": source_counts.get("Google", 0)},
        {"kennzahl": "yelp_reviews_fuer_topic_grafik", "wert": source_counts.get("Yelp", 0)},
        {"kennzahl": "differenz_referenz_minus_topic_grafik", "wert": reference["ratings_total_reference"] - len(reviews)},
    ])

    with pd.ExcelWriter(OUTPUT_XLSX) as writer:
        effects.to_excel(writer, sheet_name="topic_rating_effects", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

    plot_topic_rating_effects(effects)

    print("FERTIG")
    print(f"Referenzratings: {reference['ratings_total_reference']:,}")
    print(f"Referenzfilialen: {reference['restaurants_total_reference']:,}")
    print(f"Reviews fuer Topic-Grafik: {len(reviews):,}")
    print(f"Yelp: {source_counts.get('Yelp', 0):,}")
    print(f"Google: {source_counts.get('Google', 0):,}")
    print(f"Differenz: {reference['ratings_total_reference'] - len(reviews):,}")
    print(f"Excel: {OUTPUT_XLSX}")
    print(f"Plot: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()

