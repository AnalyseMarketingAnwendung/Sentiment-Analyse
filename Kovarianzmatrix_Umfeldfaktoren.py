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
ENRICHED_FILE = DATA_DIR / "ready_for_regression_ENRICHED_COMPLETED_with_project5_factors.csv"
TOPIC_COLS = ["FOOD", "DRINKS", "SERVICE", "SPEED", "HYGIENE", "VALUE", "AMBIENCE", "PARKING", "ACCESSIBILITY"]
EXCLUDE_ENV = {"sternrating_filiale", "anzahl_reviews", "avg_rating", "rating", "review_count", "google_review_count", "google_rating_mean", "google_text_len_mean", "share_1star", "share_2star", "share_3star", "share_4star", "share_5star", "rating_final", "state_fips", "county_fips", "tract_code", "acs_year", "lat_round", "lon_round", "phone", "attributes.waitlist_reservation", "attributes.business_temp_closed", "location.address3", "location.cross_streets", "latitude_yelp", "longitude_yelp", "coordinates.latitude", "coordinates.longitude", "business_id_project5"}

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

def load_store_ratings():
    yelp = pd.read_excel(YELP_TIGHT_FILE).rename(columns={"business_id": "store_id", "review_stars": "rating"})
    yelp["source_dataset"] = "Yelp"
    google = pd.read_excel(GOOGLE_FILE, sheet_name="Alle_Daten")
    google["source_dataset"] = "Google"
    ratings = pd.concat([yelp[["store_id", "rating", "business_name", "source_dataset"]], google[["store_id", "rating", "business_name", "source_dataset"]]], ignore_index=True)
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings[ratings["rating"].between(1, 5)].copy()
    return ratings.groupby(["source_dataset", "store_id"], dropna=False).agg(sternrating_filiale=("rating", "mean"), ratings_gesamt=("rating", "count"), filiale_name=("business_name", "first")).reset_index()

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

def environment_data(absa, store_ratings):
    enriched = pd.read_csv(ENRICHED_FILE)
    sentiments = absa.groupby("store_id", dropna=False)[TOPIC_COLS].mean().reset_index().rename(columns={topic: f"{topic}_sentiment_mean" for topic in TOPIC_COLS})
    ratings = store_ratings[["store_id", "sternrating_filiale"]].drop_duplicates("store_id")
    merged = ratings.merge(sentiments, on="store_id", how="left").merge(enriched, left_on="store_id", right_on="id", how="inner")
    env_cols = []
    for col in merged.select_dtypes(include=[np.number, bool]).columns:
        if col in EXCLUDE_ENV or "_sentiment" in col:
            continue
        series = pd.to_numeric(merged[col], errors="coerce")
        if series.notna().sum() >= 20 and series.nunique(dropna=True) > 1:
            env_cols.append(col)
    return merged, env_cols

def umfeldmatrix(absa, store_ratings):
    merged, env_cols = environment_data(absa, store_ratings)
    outcome_cols = ["sternrating_filiale"] + [f"{topic}_sentiment_mean" for topic in TOPIC_COLS]
    matrix = pd.DataFrame(index=outcome_cols, columns=env_cols, dtype=float)
    for outcome in outcome_cols:
        y = pd.to_numeric(merged[outcome], errors="coerce")
        for col in env_cols:
            x = pd.to_numeric(merged[col], errors="coerce")
            matrix.loc[outcome, col] = y.corr(x)
    display_matrix = matrix.rename(index={"sternrating_filiale": "Gesamtrating"})
    redundancy_groups = {
        "p5_tourism_level_High": "tourism_level",
        "p5_tourism_level_Low": "tourism_level",
        "p5_tourism_level_Medium": "tourism_level",
        "p5_avg_annual_temp_f": "temperature",
        "avg_temp_f": "temperature",
        "acs_black_one_race_pct": "race_ethnicity",
        "acs_white_one_race_pct": "race_ethnicity",
        "acs_non_hispanic_white_pct": "race_ethnicity",
        "acs_hispanic_any_race_pct": "race_ethnicity",
        "acs_owner_occupied_pct": "housing_tenure",
        "acs_renter_occupied_pct": "housing_tenure",
        "p5_urban_rural_Suburban": "urban_rural",
        "p5_urban_rural_Urban": "urban_rural",
        "tract_land_area_sqm": "urban_rural",
        "p5_obesity_rate_pct": "obesity",
        "obesity_rate": "obesity",
    }
    excluded_plot_factors = {
        "acs_black_one_race_pct",
        "acs_white_one_race_pct",
        "acs_non_hispanic_white_pct",
        "acs_hispanic_any_race_pct",
    }
    scores = matrix.abs().sum(axis=0).sort_values(ascending=False)
    top_cols = []
    used_groups = set()
    for col in scores.index:
        if col in excluded_plot_factors:
            continue
        group = redundancy_groups.get(col, col)
        if group in used_groups:
            continue
        top_cols.append(col)
        used_groups.add(group)
        if len(top_cols) == 12:
            break
    plot_heatmap(display_matrix[top_cols], "Kovarianzmatrix_Umfeldfaktoren", "Top-12 Umfeldfaktoren vs. Gesamtrating und Sentiments", "Korrelation")
    top = matrix.stack().reset_index()
    top.columns = ["zielvariable", "umfeldfaktor", "correlation"]
    top["abs_correlation"] = top["correlation"].abs()
    top = top.sort_values("abs_correlation", ascending=False)
    top_col_scores = scores.reset_index()
    top_col_scores.columns = ["umfeldfaktor", "sum_abs_correlation"]
    selected_top_col_scores = top_col_scores[top_col_scores["umfeldfaktor"].isin(top_cols)].copy()
    selected_top_col_scores["plot_order"] = selected_top_col_scores["umfeldfaktor"].map({col: idx + 1 for idx, col in enumerate(top_cols)})
    selected_top_col_scores = selected_top_col_scores.sort_values("plot_order")
    save_named_excel("Kovarianzmatrix_Umfeldfaktoren", {"grafikmatrix_top12": display_matrix[top_cols].reset_index(names="zielvariable"), "korrelationsmatrix_alle": matrix.reset_index(names="zielvariable"), "top_korrelationen": top, "top12_faktoren": selected_top_col_scores, "merged_filialen": merged})

if __name__ == "__main__":
    umfeldmatrix(load_absa_with_meta(), load_store_ratings())

