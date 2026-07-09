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
OPTIONS = {"delivery": [r"\bdelivery\b", r"restaurantsdelivery", r"no-contact delivery"], "takeout": [r"\btakeout\b", r"restaurantstakeout"], "dine_in": [r"\bdine-in\b", r"\bdine in\b"], "drive_through": [r"drive-through", r"drive through", r"drivethrough"], "outdoor_seating": [r"outdoor seating", r"outdoorseating"], "curbside_pickup": [r"curbside pickup"], "bike_parking": [r"bikeparking", r"bike parking"], "credit_cards": [r"businessacceptscreditcards", r"credit cards"]}

def safe_sheet_name(name):
    return re.sub(r"[\[\]:*?/\\]", "_", name)[:31]

def save_named_excel(name, sheets):
    path = DESKTOP_DIR / f"{name}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, df in sheets.items():
            df.to_excel(writer, sheet_name=safe_sheet_name(sheet), index=False)
    return path

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

if __name__ == "__main__":
    service_options_matrix(load_absa_with_meta())
