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

PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
DESKTOP_DIR = Path.home() / "Desktop" / "ChickFilA_Diagramme_Enger_Filter"
DESKTOP_DIR.mkdir(parents=True, exist_ok=True)
YELP_TIGHT_FILE = DATA_DIR / "chickfila_filtered.xlsx"
YELP_META_FILE = BASE_DIR / "servqual_outputs" / "chick_fil_a_yelp_filtered.csv"
GOOGLE_FILE = DATA_DIR / "ChickFilA_Bereinigt.xlsx"
ABSA_YELP_FILE = DATA_DIR / "absa_review_level_streng2.xlsx"
ABSA_GOOGLE_FILE = DATA_DIR / "absa_review_level_washington2.xlsx"
TIGHT_RATINGS_FILE = Path("/Users/clara/Documents/New project/outputs/chickfila_ratings/chickfila_ratings_yelp_google.xlsx")
ENRICHED_FILE = DATA_DIR / "ready_for_regression_ENRICHED_COMPLETED_with_project5_factors.csv"
USER_FEATURES_FILE = BASE_DIR / "kundenunterschiede_outputs" / "kundenunterschiede_user_features.xlsx"
USER_BIAS_FILE = BASE_DIR / "kundenunterschiede_outputs" / "user_bias_full_datasets_ohne_chickfila_min_5.xlsx"

TOPIC_COLS = ["FOOD", "DRINKS", "SERVICE", "SPEED", "HYGIENE", "VALUE", "AMBIENCE", "PARKING", "ACCESSIBILITY"]
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

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 11
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

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

def load_tight_reference_counts():
    ratings = pd.read_excel(TIGHT_RATINGS_FILE, sheet_name="Ratings")
    return {"ratings_total_reference": len(ratings), "restaurants_total_reference": ratings["Restaurant"].nunique(dropna=True), "google_ratings_reference": int((ratings["Quelle"] == "Google").sum()), "yelp_ratings_reference": int((ratings["Quelle"] == "Yelp").sum())}

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

def compile_patterns(keyword_dict):
    patterns = {}
    for key, words in keyword_dict.items():
        escaped = [re.escape(str(word).lower()).replace(r"\ ", r"\s+") for word in words]
        patterns[key] = re.compile(r"\b(?:" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)
    return patterns

WORD_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z'-]*\b")

def count_words(value):
    if pd.isna(value):
        return 0
    return len(WORD_PATTERN.findall(str(value)))

def count_mentions(value, pattern):
    if pd.isna(value):
        return 0
    return len(pattern.findall(str(value)))

def approximate_utc_offset_from_lon(lon):
    if pd.isna(lon):
        return 0
    lon = float(lon)
    if lon <= -135:
        return -10
    if lon <= -114:
        return -8
    if lon <= -101:
        return -7
    if lon <= -85:
        return -6
    return -5

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

OPTIONS = {"delivery": [r"\bdelivery\b", r"restaurantsdelivery", r"no-contact delivery"], "takeout": [r"\btakeout\b", r"restaurantstakeout"], "dine_in": [r"\bdine-in\b", r"\bdine in\b"], "drive_through": [r"drive-through", r"drive through", r"drivethrough"], "outdoor_seating": [r"outdoor seating", r"outdoorseating"], "curbside_pickup": [r"curbside pickup"], "bike_parking": [r"bikeparking", r"bike parking"], "credit_cards": [r"businessacceptscreditcards", r"credit cards"]}

def has_option(value, patterns):
    if pd.isna(value):
        return 0
    text = str(value).lower()
    return int(any(re.search(pattern, text) for pattern in patterns))

def plot_heatmap(matrix, name, title, cbar_label, vmin=-1, vmax=1):
    values = matrix.to_numpy(dtype=float)
    if vmin is None or vmax is None:
        vmax_abs = np.nanmax(np.abs(values))
        vmin, vmax = -vmax_abs, vmax_abs
    fig, ax = plt.subplots(figsize=(13.5, 7))
    im = ax.imshow(values, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label(cbar_label, fontweight="bold")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if pd.notna(value):
                color = "white" if abs(value) >= 0.55 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color, fontweight="bold")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(DESKTOP_DIR / f"{name}.png", dpi=250)
    plt.close(fig)

def service_options_matrix(absa):
    reviews = absa.copy()
    for option, patterns in OPTIONS.items():
        reviews[option] = reviews["service_options"].map(lambda value: has_option(value, patterns))
    agg = {"rating": "mean", "business_name": "first", "source_dataset": "first"}
    for topic in TOPIC_COLS:
        agg[topic] = "mean"
    for option in OPTIONS:
        agg[option] = "max"
    store_table = reviews.groupby("store_id", dropna=False).agg(agg).reset_index().rename(columns={"rating": "avg_rating"})
    rows = []
    for option in OPTIONS:
        for metric in ["avg_rating"] + TOPIC_COLS:
            values = pd.to_numeric(store_table[metric], errors="coerce")
            with_mask = store_table[option].eq(1) & values.notna()
            without_mask = store_table[option].eq(0) & values.notna()
            rows.append({"service_option": option, "metric": metric, "n_filialen_with_option": int(with_mask.sum()), "mean_with_option": values[with_mask].mean(), "n_filialen_without_option": int(without_mask.sum()), "mean_without_option": values[without_mask].mean(), "difference_with_minus_without": values[with_mask].mean() - values[without_mask].mean()})
    diff = pd.DataFrame(rows)
    matrix = diff.pivot(index="service_option", columns="metric", values="difference_with_minus_without").reindex(columns=["avg_rating"] + TOPIC_COLS)
    plot_heatmap(matrix, "Kovarianzmatrix_Serviceoptionen", "Service Options: Rating- und Topic-Differenzen auf Filialebene", "Differenz: vorhanden minus nicht vorhanden")
    save_named_excel("Kovarianzmatrix_Serviceoptionen", {"differenzen": diff, "matrix": matrix.reset_index(), "filialen": store_table})

def kundenmatrix():
    features = pd.read_excel(USER_FEATURES_FILE)
    row_vars = ["review_count_dataset", "reviewed_chickfila_more_than_once", "avg_review_word_count", "total_review_words", "unique_store_count", "avg_local_hour", "weekend_share", "daypart_morning_share", "daypart_afternoon_share", "daypart_evening_share", "daypart_night_share", "gender_female", "gender_male", "yelp_user_review_count", "yelp_user_average_stars", "avg_rating_without_chickfila"]
    col_vars = ["avg_star_rating"] + [f"{topic}_sentiment_mean" for topic in TOPIC_COLS]
    row_vars = [col for col in row_vars if col in features.columns]
    col_vars = [col for col in col_vars if col in features.columns]
    matrix = pd.DataFrame(index=row_vars, columns=col_vars, dtype=float)
    for row in row_vars:
        for col in col_vars:
            matrix.loc[row, col] = pd.to_numeric(features[row], errors="coerce").corr(pd.to_numeric(features[col], errors="coerce"))
    plot_heatmap(matrix, "Kovarianzmatrix_Kundeneigenschaften", "Kundenbeschreibung vs. Rating und Sentiments", "Korrelation")
    save_named_excel("Kovarianzmatrix_Kundeneigenschaften", {"korrelationsmatrix": matrix.reset_index(names="kundenbeschreibung"), "user_features": features})

def userrating_verteilung():
    users = pd.read_excel(USER_BIAS_FILE)
    fig, ax = plt.subplots(figsize=(9, 5.8))
    bins = np.arange(1, 5.25, 0.25)
    yelp = users.loc[users["source_dataset"].eq("yelp"), "avg_rating_without_chickfila"]
    google = users.loc[users["source_dataset"].eq("google"), "avg_rating_without_chickfila"]
    ax.hist([yelp, google], bins=bins, stacked=True, color=["#b71c1c", "#f9caca"], edgecolor="black", label=["Yelp", "Google"])
    ax.axvline(users["avg_rating_without_chickfila"].mean(), color="black", linewidth=2)
    ax.set_title("User-Durchschnittsrating ohne Chick-fil-A, min. 5 Reviews")
    ax.set_xlabel("Durchschnittsrating je User ohne Chick-fil-A")
    ax.set_ylabel("Anzahl User")
    style_ax(ax)
    legend = ax.legend(frameon=True, facecolor="white", edgecolor="black")
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(DESKTOP_DIR / "Verteilung_Userratings_Mehrfachbewerter.png", dpi=250)
    plt.close(fig)
    summary = users.groupby("source_dataset").agg(users=("global_user_id", "count"), avg_rating=("avg_rating_without_chickfila", "mean"), median_rating=("avg_rating_without_chickfila", "median")).reset_index()
    save_named_excel("Verteilung_Userratings_Mehrfachbewerter", {"userratings": users, "summary": summary})

def boxplots_bewertungszeitpunkt(reviews):
    df = reviews.copy()
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["time_utc"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    offsets = df["longitude"].map(approximate_utc_offset_from_lon)
    df["local_time_approx"] = df["time_utc"] + pd.to_timedelta(offsets, unit="h")
    df["is_weekend"] = df["local_time_approx"].dt.weekday.isin([5, 6])
    df["wochenende"] = np.where(df["is_weekend"], "Wochenende", "Wochentag")
    df["is_evening"] = df["local_time_approx"].dt.hour.between(18, 23)
    df["abend"] = np.where(df["is_evening"], "Abend", "Rest des Tages")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    for ax, col, order, title in [(axes[0], "wochenende", ["Wochentag", "Wochenende"], "Wochentag vs. Wochenende"), (axes[1], "abend", ["Rest des Tages", "Abend"], "Abend vs. Rest des Tages")]:
        data = [df.loc[df[col].eq(label), "rating"] for label in order]
        box = ax.boxplot(data, labels=order, patch_artist=True, showmeans=True, medianprops={"color": "black", "linewidth": 2}, meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 7})
        for patch, color in zip(box["boxes"], ["#f9caca", "#b71c1c"]):
            patch.set_facecolor(color)
        ax.set_title(title)
        ax.set_ylabel("Kundenrating")
        style_ax(ax)
    fig.tight_layout()
    fig.savefig(DESKTOP_DIR / "Boxplots_Bewertungszeitpunkt.png", dpi=250)
    plt.close(fig)
    summary = pd.concat([df.groupby("wochenende").agg(reviews=("rating", "count"), mean_rating=("rating", "mean"), median_rating=("rating", "median")).reset_index().rename(columns={"wochenende": "gruppe"}), df.groupby("abend").agg(reviews=("rating", "count"), mean_rating=("rating", "mean"), median_rating=("rating", "median")).reset_index().rename(columns={"abend": "gruppe"})], ignore_index=True)
    export_df = df.copy()
    for col in export_df.select_dtypes(include=["datetimetz"]).columns:
        export_df[col] = export_df[col].dt.tz_localize(None)
    save_named_excel("Boxplots_Bewertungszeitpunkt", {"ratings_mit_zeit": export_df, "summary": summary})

EXCLUDE_ENV = {"sternrating_filiale", "anzahl_reviews", "avg_rating", "rating", "review_count", "google_review_count", "google_rating_mean", "google_text_len_mean", "share_1star", "share_2star", "share_3star", "share_4star", "share_5star", "rating_final", "state_fips", "county_fips", "tract_code", "acs_year", "lat_round", "lon_round", "phone", "attributes.waitlist_reservation", "attributes.business_temp_closed", "location.address3", "location.cross_streets", "latitude_yelp", "longitude_yelp", "coordinates.latitude", "coordinates.longitude", "business_id_project5"}

def environment_data(absa, store_ratings):
    enriched = pd.read_csv(ENRICHED_FILE)
    sentiments = absa.groupby("store_id", dropna=False)[TOPIC_COLS].mean().reset_index().rename(columns={topic: f"{topic}_sentiment_mean" for topic in TOPIC_COLS})
    ratings = store_ratings[["store_id", "sternrating_filiale"]].drop_duplicates("store_id")
    merged = ratings.merge(sentiments, on="store_id", how="left").merge(enriched, left_on="store_id", right_on="id", how="inner")
    env_cols = []
    for col in merged.select_dtypes(include=[np.number, bool]).columns:
        if col in EXCLUDE_ENV or col.endswith("_sentiment"):
            continue
        series = pd.to_numeric(merged[col], errors="coerce")
        if series.notna().sum() >= 20 and series.nunique(dropna=True) > 1:
            env_cols.append(col)
    return merged, env_cols

def umfeldmatrix(absa, store_ratings):
    merged, env_cols = environment_data(absa, store_ratings)
    sent_cols = [f"{topic}_sentiment_mean" for topic in TOPIC_COLS]
    matrix = pd.DataFrame(index=sent_cols, columns=env_cols, dtype=float)
    for sent in sent_cols:
        y = pd.to_numeric(merged[sent], errors="coerce")
        for col in env_cols:
            x = pd.to_numeric(merged[col], errors="coerce")
            matrix.loc[sent, col] = y.corr(x)
    top_cols = matrix.abs().max(axis=0).sort_values(ascending=False).head(25).index.tolist()
    plot_heatmap(matrix[top_cols], "Kovarianzmatrix_Umfeldfaktoren", "Standort-/Umfeldfaktoren vs. Sentiments", "Korrelation")
    top = matrix.stack().reset_index()
    top.columns = ["sentiment", "umfeldfaktor", "correlation"]
    top["abs_correlation"] = top["correlation"].abs()
    top = top.sort_values("abs_correlation", ascending=False)
    save_named_excel("Kovarianzmatrix_Umfeldfaktoren", {"korrelationsmatrix": matrix.reset_index(names="sentiment"), "top_korrelationen": top, "merged_filialen": merged})

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot else np.nan

def ridge_fit(x, y, alpha):
    x_aug = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(x_aug.shape[1]) * alpha
    penalty[0, 0] = 0
    return np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y)

def ridge_predict(x, beta):
    return np.column_stack([np.ones(len(x)), x]) @ beta

def regression_umfeld(absa, store_ratings):
    merged, env_cols = environment_data(absa, store_ratings)
    model = merged[["store_id", "sternrating_filiale"] + env_cols].dropna(subset=["sternrating_filiale"]).copy()
    for col in env_cols:
        model[col] = pd.to_numeric(model[col], errors="coerce").fillna(pd.to_numeric(model[col], errors="coerce").median())
    xdf = model[env_cols]
    y = model["sternrating_filiale"].to_numpy(dtype=float)
    x = xdf.to_numpy(dtype=float)
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds[stds == 0] = 1
    xs = (x - means) / stds
    rng = np.random.default_rng(42)
    folds = np.array_split(rng.permutation(len(y)), 5)
    alphas = [0.01, 0.1, 1, 3, 10, 30, 100, 300, 1000]
    cv_rows = []
    pred_by_alpha = {}
    for alpha in alphas:
        pred = np.full(len(y), np.nan)
        for test_idx in folds:
            train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
            beta = ridge_fit(xs[train_idx], y[train_idx], alpha)
            pred[test_idx] = ridge_predict(xs[test_idx], beta)
        cv_rows.append({"alpha": alpha, "cv_r2": r2_score(y, pred)})
        pred_by_alpha[alpha] = pred
    cv = pd.DataFrame(cv_rows).sort_values("cv_r2", ascending=False)
    best_alpha = float(cv.iloc[0]["alpha"])
    cv_pred = pred_by_alpha[best_alpha]
    beta = ridge_fit(xs, y, best_alpha)
    full_pred = ridge_predict(xs, beta)
    coef = pd.DataFrame({"feature": env_cols, "standardized_coefficient": beta[1:], "abs_standardized_coefficient": np.abs(beta[1:])}).sort_values("abs_standardized_coefficient", ascending=False)
    summary = pd.DataFrame([{"metric": "n_filialen", "value": len(y)}, {"metric": "n_umfeldfaktoren", "value": len(env_cols)}, {"metric": "best_alpha", "value": best_alpha}, {"metric": "cv_r2", "value": r2_score(y, cv_pred)}, {"metric": "in_sample_r2", "value": r2_score(y, full_pred)}])
    predictions = model[["store_id", "sternrating_filiale"]].copy()
    predictions["ridge_cv_prediction"] = cv_pred
    predictions["ridge_in_sample_prediction"] = full_pred
    top = coef.head(12).sort_values("standardized_coefficient")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(y, cv_pred, color="#b71c1c", edgecolor="black", alpha=0.75)
    lims = [min(y.min(), np.nanmin(cv_pred)), max(y.max(), np.nanmax(cv_pred))]
    axes[0].plot(lims, lims, color="black", linewidth=2)
    axes[0].set_title("Vorhersage vs. tatsächliches Rating")
    axes[0].set_xlabel("tatsächliches Filialrating")
    axes[0].set_ylabel("CV-Vorhersage")
    axes[0].text(0.04, 0.96, f"CV-R² = {r2_score(y, cv_pred):.3f}\nIn-sample R² = {r2_score(y, full_pred):.3f}\nn = {len(y)}", transform=axes[0].transAxes, ha="left", va="top", bbox={"facecolor": "white", "edgecolor": "black", "pad": 6}, fontweight="bold")
    style_ax(axes[0])
    colors = np.where(top["standardized_coefficient"] >= 0, "#b71c1c", "#222222")
    axes[1].barh(np.arange(len(top)), top["standardized_coefficient"], color=colors, edgecolor="black")
    axes[1].set_yticks(np.arange(len(top)))
    axes[1].set_yticklabels(top["feature"], fontsize=8)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title("Stärkste Ridge-Koeffizienten")
    axes[1].set_xlabel("standardisierter Koeffizient")
    style_ax(axes[1])
    fig.suptitle("Erklärung des Filialratings durch Umfeldfaktoren", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(DESKTOP_DIR / "Regression_Einfluss_Umfeldfaktoren.png", dpi=250)
    plt.close(fig)
    save_named_excel("Regression_Einfluss_Umfeldfaktoren", {"summary": summary, "cv_alphas": cv, "koeffizienten": coef, "predictions": predictions, "modelldaten": model})

if __name__ == "__main__":
    kundenmatrix()
