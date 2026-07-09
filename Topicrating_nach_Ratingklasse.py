from pathlib import Path
import os
import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
DESKTOP_DIR = Path.home() / "Desktop" / "ChickFilA_Diagramme_Enger_Filter"
YELP_TIGHT_FILE = DATA_DIR / "chickfila_filtered.xlsx"
YELP_META_FILE = BASE_DIR / "servqual_outputs" / "chick_fil_a_yelp_filtered.csv"
GOOGLE_FILE = DATA_DIR / "ChickFilA_Bereinigt.xlsx"
ABSA_YELP_FILE = DATA_DIR / "absa_review_level_streng2.xlsx"
ABSA_GOOGLE_FILE = DATA_DIR / "absa_review_level_washington2.xlsx"
TOPIC_COLS = ["FOOD", "DRINKS", "SERVICE", "SPEED", "HYGIENE", "VALUE", "AMBIENCE", "PARKING", "ACCESSIBILITY"]
RATING_CLASS_LABELS = ["1.0-1.9", "2.0-2.9", "3.0-3.9", "4.0-4.4", "4.5-5.0"]
RED_PALETTE = ["#fff5f5", "#f9caca", "#f29a9a", "#e65f5f", "#c92a2a", "#a51111", "#7f0000", "#4d0000", "#111111", "#000000"]

def safe_sheet_name(name):
    return re.sub(r"[\[\]:*?/\\]", "_", name)[:31]

def save_named_excel(name, sheets):
    path = DESKTOP_DIR / f"{name}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, df in sheets.items():
            df.to_excel(writer, sheet_name=safe_sheet_name(sheet), index=False)
    return path

def style_ax(ax):
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")
    ax.grid(False)
    ax.tick_params(colors="black")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_color("black")

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

def load_yelp_absa_reviews():
    absa = pd.read_excel(ABSA_YELP_FILE)
    source = pd.read_excel(YELP_TIGHT_FILE).rename(columns={"business_id": "store_id", "review_stars": "rating", "review_date": "time"})
    absa["time_key"] = pd.to_datetime(absa["time"], errors="coerce")
    source["time_key"] = pd.to_datetime(source["time"], errors="coerce")
    merged = absa.merge(source[["user_id", "store_id", "time_key", "rating", "business_name"]], on=["user_id", "store_id", "time_key"], how="inner")
    merged["source_dataset"] = "Yelp"
    return merged

def load_google_absa_reviews():
    absa = pd.read_excel(ABSA_GOOGLE_FILE)
    source = pd.read_excel(GOOGLE_FILE, sheet_name="Alle_Daten")
    absa["user_prefix6"] = absa["user_id"].map(user_prefix)
    source["user_prefix6"] = source["user_id"].map(user_prefix)
    absa["time_key"] = pd.to_datetime(absa["time"], errors="coerce").dt.floor("s")
    source["time_key"] = pd.to_datetime(source["time"], errors="coerce").dt.floor("s")
    key_cols = ["user_prefix6", "store_id", "time_key"]
    absa = absa.groupby(key_cols, dropna=False, as_index=False).agg({**{topic: "mean" for topic in TOPIC_COLS}, "user_id": "first", "time": "first"})
    merged = absa.merge(source[key_cols + ["rating", "business_name"]], on=key_cols, how="inner")
    merged["source_dataset"] = "Google"
    return merged

def prepare_reviews():
    reviews = pd.concat([load_yelp_absa_reviews(), load_google_absa_reviews()], ignore_index=True, sort=False)
    reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")
    reviews = reviews[reviews["rating"].between(1, 5)].copy()
    for topic in TOPIC_COLS:
        reviews[topic] = pd.to_numeric(reviews[topic], errors="coerce")
        reviews[f"{topic}_mentioned"] = reviews[topic].notna().astype(int)
    return reviews

def load_tight_rating_reviews():
    yelp = pd.read_excel(YELP_TIGHT_FILE).rename(columns={"business_id": "store_id", "review_stars": "rating", "review_date": "time"})
    yelp["source_dataset"] = "Yelp"
    google = pd.read_excel(GOOGLE_FILE, sheet_name="Alle_Daten")
    google["source_dataset"] = "Google"
    cols = ["source_dataset", "user_id", "store_id", "time", "rating", "text", "business_name"]
    optional = ["address", "latitude", "longitude", "service_options"]
    for col in optional:
        if col not in yelp.columns:
            yelp[col] = np.nan
        if col not in google.columns:
            google[col] = np.nan
    reviews = pd.concat([yelp[cols + optional], google[cols + optional]], ignore_index=True)
    reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")
    return reviews[reviews["rating"].between(1, 5)].copy()

def enrich_reviews_with_service_options(reviews):
    reviews = reviews.copy()
    meta = pd.read_csv(YELP_META_FILE, usecols=["user_id", "time", "store_id", "service_options", "address", "latitude", "longitude"])
    reviews["time_key"] = pd.to_datetime(reviews["time"], errors="coerce")
    meta["time_key"] = pd.to_datetime(meta["time"], errors="coerce")
    yelp_mask = reviews["source_dataset"].eq("Yelp")
    yelp = reviews.loc[yelp_mask].drop(columns=["service_options", "address", "latitude", "longitude"], errors="ignore").merge(meta.drop(columns=["time"]), on=["user_id", "store_id", "time_key"], how="left")
    google = reviews.loc[~yelp_mask].copy()
    return pd.concat([yelp, google], ignore_index=True, sort=False).drop(columns=["time_key"], errors="ignore")

def load_absa_with_meta():
    absa = prepare_reviews()
    ratings = enrich_reviews_with_service_options(load_tight_rating_reviews())
    ratings["time_key"] = pd.to_datetime(ratings["time"], errors="coerce").dt.floor("s")
    absa["time_key"] = pd.to_datetime(absa["time_key"], errors="coerce").dt.floor("s")
    yelp_absa = absa[absa["source_dataset"].eq("Yelp")].merge(ratings[ratings["source_dataset"].eq("Yelp")][["source_dataset", "user_id", "store_id", "time_key", "service_options", "address", "latitude", "longitude"]], on=["source_dataset", "user_id", "store_id", "time_key"], how="left")
    google_absa = absa[absa["source_dataset"].eq("Google")].merge(ratings[ratings["source_dataset"].eq("Google")][["source_dataset", "store_id", "time_key", "service_options", "address", "latitude", "longitude"]], on=["source_dataset", "store_id", "time_key"], how="left")
    return pd.concat([yelp_absa, google_absa], ignore_index=True, sort=False)

def load_store_ratings():
    yelp = pd.read_excel(YELP_TIGHT_FILE).rename(columns={"business_id": "store_id", "review_stars": "rating"})
    yelp["source_dataset"] = "Yelp"
    google = pd.read_excel(GOOGLE_FILE, sheet_name="Alle_Daten")
    google["source_dataset"] = "Google"
    ratings = pd.concat([yelp[["store_id", "rating", "business_name", "source_dataset"]], google[["store_id", "rating", "business_name", "source_dataset"]]], ignore_index=True)
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings[ratings["rating"].between(1, 5)].copy()
    return ratings.groupby(["source_dataset", "store_id"], dropna=False).agg(sternrating_filiale=("rating", "mean"), ratings_gesamt=("rating", "count"), filiale_name=("business_name", "first")).reset_index()

def add_rating_class(df):
    df = df.copy()
    df["rating_klasse"] = pd.cut(df["sternrating_filiale"], bins=[1.0, 2.0, 3.0, 4.0, 4.5, 5.0], labels=RATING_CLASS_LABELS, include_lowest=True, right=False)
    df.loc[df["sternrating_filiale"] == 5.0, "rating_klasse"] = RATING_CLASS_LABELS[-1]
    return df

def plot_grouped(df, x_col, value_cols, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(df[x_col]))
    width = 0.75 / max(len(value_cols), 1)
    for idx, col in enumerate(value_cols):
        offset = (idx - (len(value_cols) - 1) / 2) * width
        ax.bar(x + offset, df[col], width=width, label=col.replace("_sentiment_mean", ""), color=RED_PALETTE[idx % len(RED_PALETTE)], edgecolor="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("Filial-Ratingklasse")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col])
    style_ax(ax)
    legend = ax.legend(frameon=True, facecolor="white", edgecolor="black", loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(path, dpi=250)
    plt.close(fig)

def topicrating_nach_ratingklasse(absa, store_ratings):
    store_sent = absa.groupby(["source_dataset", "store_id"], dropna=False)[TOPIC_COLS].mean().reset_index()
    store_table = add_rating_class(store_ratings.merge(store_sent, on=["source_dataset", "store_id"], how="left"))
    class_table = store_table.groupby("rating_klasse", observed=False)[TOPIC_COLS].mean().reindex(RATING_CLASS_LABELS).reset_index()
    plot_df = class_table.rename(columns={topic: f"{topic}_sentiment_mean" for topic in TOPIC_COLS})
    plot_grouped(plot_df, "rating_klasse", [f"{topic}_sentiment_mean" for topic in TOPIC_COLS], "Durchschnittliches Sentiment-Rating nach Filial-Ratingklasse", "Durchschnittliches Sentiment-Rating", DESKTOP_DIR / "Topicrating_nach_Ratingklasse.png")
    save_named_excel("Topicrating_nach_Ratingklasse", {"ratingklassen": class_table, "filialen": store_table})

if __name__ == "__main__":
    topicrating_nach_ratingklasse(load_absa_with_meta(), load_store_ratings())

