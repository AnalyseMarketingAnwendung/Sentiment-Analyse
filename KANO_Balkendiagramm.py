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
GOOGLE_FILE = DATA_DIR / "ChickFilA_Bereinigt.xlsx"
RATING_CLASS_LABELS = ["1.0-1.9", "2.0-2.9", "3.0-3.9", "4.0-4.4", "4.5-5.0"]
RED_PALETTE = ["#fff5f5", "#f9caca", "#f29a9a", "#e65f5f", "#c92a2a", "#a51111", "#7f0000", "#4d0000", "#111111", "#000000"]
KANO_KEYWORDS = {'BASISMERKMALE': ['unfortunately',
                   'sadly',
                   'regrettably',
                   'disappointed',
                   'disappointing',
                   'disappointment',
                   'frustrated',
                   'frustrating',
                   'upset',
                   'annoyed',
                   'annoying',
                   'unacceptable',
                   'not acceptable',
                   'not okay',
                   'not good',
                   'not right',
                   'not worth it',
                   'should have',
                   'should be',
                   'should not',
                   'expected better',
                   'expected more',
                   'below expectations',
                   'let down',
                   'let me down',
                   'failed',
                   'failure',
                   'poor',
                   'terrible',
                   'awful',
                   'horrible',
                   'bad',
                   'worst',
                   'ridiculous',
                   'never again',
                   'waste',
                   'waste of money',
                   'problem',
                   'issue',
                   'complaint',
                   'unhappy',
                   'dissatisfied'],
 'BEGEISTERUNGSMERKMALE': ['surprisingly',
                           'surprised',
                           'surprising',
                           'unexpectedly',
                           'unexpected',
                           'luckily',
                           'fortunate',
                           'fortunately',
                           'delighted',
                           'delightful',
                           'impressed',
                           'impressive',
                           'amazed',
                           'amazing',
                           'wow',
                           'wonderful',
                           'fantastic',
                           'exceptional',
                           'outstanding',
                           'excellent',
                           'awesome',
                           'incredible',
                           'unbelievable',
                           'perfect',
                           'remarkable',
                           'memorable',
                           'special',
                           'pleasantly surprised',
                           'above and beyond',
                           'extra mile',
                           'made my day',
                           'exceeded expectations',
                           'better than expected',
                           'more than expected'],
 'LEISTUNGSMERKMALE': ['better',
                       'best',
                       'worse',
                       'worst',
                       'faster',
                       'fastest',
                       'quicker',
                       'quickest',
                       'slower',
                       'slowest',
                       'higher',
                       'highest',
                       'lower',
                       'lowest',
                       'more',
                       'less',
                       'improved',
                       'improvement',
                       'improving',
                       'declined',
                       'decline',
                       'worsened',
                       'consistent',
                       'more consistent',
                       'less consistent',
                       'efficient',
                       'more efficient',
                       'less efficient',
                       'worth',
                       'worth it',
                       'better value',
                       'better quality',
                       'higher quality',
                       'lower quality',
                       'better service',
                       'faster service',
                       'quicker service',
                       'better than',
                       'worse than',
                       'compared to',
                       'as good as',
                       'not as good',
                       'outperformed',
                       'underperformed',
                       'exceeded',
                       'met expectations']}
WORD_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z'-]*\b")

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

def compile_patterns(keyword_dict):
    patterns = {}
    for key, words in keyword_dict.items():
        escaped = [re.escape(str(word).lower()).replace(r"\ ", r"\s+") for word in words]
        patterns[key] = re.compile(r"\b(?:" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)
    return patterns

def count_words(value):
    if pd.isna(value):
        return 0
    return len(WORD_PATTERN.findall(str(value)))

def count_mentions(value, pattern):
    if pd.isna(value):
        return 0
    return len(pattern.findall(str(value)))

def keyword_ratingklasse(name, reviews, store_ratings, keyword_dict):
    patterns = compile_patterns(keyword_dict)
    df = reviews.copy()
    df["word_count"] = df["text"].map(count_words)
    for category, pattern in patterns.items():
        df[f"{category}_mentions"] = df["text"].map(lambda value: count_mentions(value, pattern))
    review_agg = df.groupby(["source_dataset", "store_id"], dropna=False).agg(words=("word_count", "sum"), reviews=("rating", "count"), **{f"{category}_mentions": (f"{category}_mentions", "sum") for category in keyword_dict}).reset_index()
    store_table = add_rating_class(store_ratings.merge(review_agg, on=["source_dataset", "store_id"], how="left").fillna(0))
    class_table = store_table.groupby("rating_klasse", observed=False).agg(filialen=("store_id", "count"), ratings_gesamt=("ratings_gesamt", "sum"), words=("words", "sum"), **{f"{category}_mentions": (f"{category}_mentions", "sum") for category in keyword_dict}).reindex(RATING_CLASS_LABELS).fillna(0).reset_index()
    value_cols = []
    for category in keyword_dict:
        col = f"{category}_pro_1000_woerter"
        class_table[col] = np.where(class_table["words"] > 0, class_table[f"{category}_mentions"] / class_table["words"] * 1000, 0)
        value_cols.append(col)
    fig, ax = plt.subplots(figsize=(10.5, 6))
    x = np.arange(len(class_table["rating_klasse"]))
    width = 0.75 / len(value_cols)
    for idx, col in enumerate(value_cols):
        ax.bar(x + (idx - (len(value_cols) - 1) / 2) * width, class_table[col], width=width, label=col.replace("_pro_1000_woerter", ""), color=RED_PALETTE[idx + 2], edgecolor="black", linewidth=0.6)
    ax.set_title(name.replace("_", " "))
    ax.set_xlabel("Filial-Ratingklasse")
    ax.set_ylabel("Erwähnungen pro 1000 Wörter")
    ax.set_xticks(x)
    ax.set_xticklabels(class_table["rating_klasse"])
    style_ax(ax)
    legend = ax.legend(frameon=True, facecolor="white", edgecolor="black", loc="center left", bbox_to_anchor=(1.02, 0.5))
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(DESKTOP_DIR / f"{name}.png", dpi=250)
    plt.close(fig)
    save_named_excel(name, {"ratingklassen": class_table, "filialen": store_table})

if __name__ == "__main__":
    keyword_ratingklasse("KANO_Balkendiagramm", load_tight_rating_reviews(), load_store_ratings(), KANO_KEYWORDS)
