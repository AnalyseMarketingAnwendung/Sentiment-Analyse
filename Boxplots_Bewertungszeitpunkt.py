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

def enrich_reviews_with_service_options(reviews):
    reviews = reviews.copy()
    meta = pd.read_csv(YELP_META_FILE, usecols=["user_id", "time", "store_id", "service_options", "address", "latitude", "longitude"])
    reviews["time_key"] = pd.to_datetime(reviews["time"], errors="coerce")
    meta["time_key"] = pd.to_datetime(meta["time"], errors="coerce")
    yelp_mask = reviews["source_dataset"].eq("Yelp")
    yelp = reviews.loc[yelp_mask].drop(columns=["service_options", "address", "latitude", "longitude"], errors="ignore").merge(meta.drop(columns=["time"]), on=["user_id", "store_id", "time_key"], how="left")
    google = reviews.loc[~yelp_mask].copy()
    return pd.concat([yelp, google], ignore_index=True, sort=False).drop(columns=["time_key"], errors="ignore")

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

if __name__ == "__main__":
    boxplots_bewertungszeitpunkt(enrich_reviews_with_service_options(load_tight_rating_reviews()))
