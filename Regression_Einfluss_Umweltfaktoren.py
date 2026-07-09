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
    regression_umfeld(load_absa_with_meta(), load_store_ratings())

