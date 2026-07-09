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
SERVQUAL_KEYWORDS = {'ASSURANCE': ['professional',
               'unprofessional',
               'polite',
               'rude',
               'respectful',
               'disrespectful',
               'courteous',
               'uncourteous',
               'manager',
               'management',
               'supervisor',
               'employee',
               'employees',
               'staff',
               'cashier',
               'knowledgeable',
               'trained',
               'training',
               'competent',
               'incompetent',
               'confidence',
               'confident',
               'trust',
               'trusted',
               'trustworthy',
               'safe',
               'safety',
               'careful',
               'careless',
               'attentive',
               'attention',
               'handled',
               'handled well',
               'resolved',
               'solution',
               'problem solved',
               'apologized',
               'apology',
               'explained',
               'explain',
               'answer',
               'answered',
               'question',
               'questions',
               'attitude',
               'respect',
               'security',
               'food safety',
               'allergy',
               'allergies',
               'gluten',
               'sanitized',
               'mask',
               'gloves',
               'receipt',
               'policy'],
 'EMPATHY': ['friendly',
             'friendliest',
             'kind',
             'kindness',
             'nice',
             'sweet',
             'pleasant',
             'welcoming',
             'welcome',
             'warm',
             'caring',
             'care',
             'patient',
             'patience',
             'understanding',
             'helpful',
             'help',
             'smile',
             'smiles',
             'smiling',
             'greeted',
             'greeting',
             'hello',
             'thank you',
             'thanks',
             'listened',
             'listening',
             'accommodating',
             'accommodated',
             'personal',
             'personable',
             'customer',
             'customers',
             'customer service',
             'kids',
             'children',
             'family',
             'families',
             'special request',
             'request',
             'needs',
             'attention',
             'checked on',
             'went above',
             'above and beyond',
             'extra mile',
             'made my day',
             'apologized',
             'apology',
             'sorry',
             'compassion',
             'generous',
             'hospitality',
             'treated me',
             'treated us',
             'felt welcome'],
 'RELIABILITY': ['wrong order',
                 'incorrect order',
                 'order wrong',
                 'order was wrong',
                 'messed up order',
                 'messed up',
                 'mistake',
                 'mistakes',
                 'error',
                 'incorrect',
                 'accurate',
                 'accuracy',
                 'accurate order',
                 'missing item',
                 'missing items',
                 'forgot',
                 'forgotten',
                 'forgot my',
                 'left out',
                 'did not receive',
                 'never received',
                 'no sauce',
                 'missing sauce',
                 'wrong sauce',
                 'wrong drink',
                 'wrong sandwich',
                 'wrong meal',
                 'wrong food',
                 'wrong bag',
                 'wrong receipt',
                 'charged wrong',
                 'overcharged',
                 'undercharged',
                 'refund',
                 'consistent',
                 'consistently',
                 'inconsistent',
                 'reliable',
                 'unreliable',
                 'dependable',
                 'always',
                 'never',
                 'every time',
                 'again',
                 'same issue',
                 'same problem',
                 'quality',
                 'fresh',
                 'cold food',
                 'hot food',
                 'stale',
                 'undercooked',
                 'overcooked',
                 'burnt',
                 'soggy',
                 'dry',
                 'made correctly',
                 'prepared correctly'],
 'RESPONSIVENESS': ['fast',
                    'faster',
                    'fastest',
                    'quick',
                    'quickly',
                    'slow',
                    'slowly',
                    'speed',
                    'speedy',
                    'efficient',
                    'inefficient',
                    'prompt',
                    'immediate',
                    'immediately',
                    'wait',
                    'waiting',
                    'waited',
                    'wait time',
                    'long wait',
                    'short wait',
                    'line',
                    'long line',
                    'queue',
                    'rush',
                    'busy',
                    'delay',
                    'delayed',
                    'late',
                    'took forever',
                    'took too long',
                    'took a while',
                    'ready',
                    'ready fast',
                    'order ready',
                    'served quickly',
                    'service speed',
                    'drive thru line',
                    'drive-thru line',
                    'mobile order',
                    'pickup',
                    'curbside',
                    'delivery',
                    'app order',
                    'online order',
                    'helped',
                    'helpful',
                    'assistance',
                    'assist',
                    'responded',
                    'response',
                    'ignored',
                    'ignore',
                    'waiting for help',
                    'no one helped',
                    'opened another register',
                    'register',
                    'cashier',
                    'counter'],
 'TANGIBLES': ['clean',
               'cleanliness',
               'dirty',
               'messy',
               'filthy',
               'spotless',
               'sanitary',
               'unsanitary',
               'hygiene',
               'sticky',
               'smell',
               'odor',
               'trash',
               'garbage',
               'bathroom',
               'bathrooms',
               'restroom',
               'restrooms',
               'table',
               'tables',
               'floor',
               'floors',
               'counter',
               'counters',
               'seat',
               'seats',
               'seating',
               'booth',
               'booths',
               'dining room',
               'inside',
               'interior',
               'decor',
               'lighting',
               'music',
               'atmosphere',
               'environment',
               'vibe',
               'location',
               'building',
               'restaurant',
               'parking',
               'parking lot',
               'parking spot',
               'drive thru',
               'drive-thru',
               'drive through',
               'window',
               'menu board',
               'speaker',
               'sign',
               'signage',
               'line',
               'lane',
               'lanes',
               'crowded',
               'comfortable',
               'uncomfortable',
               'renovated',
               'modern',
               'old',
               'new',
               'equipment']}
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
    keyword_ratingklasse("SERVQUAL_Balkendiagramm", load_tight_rating_reviews(), load_store_ratings(), SERVQUAL_KEYWORDS)

