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

DESKTOP_DIR = Path.home() / "Desktop" / "ChickFilA_Diagramme_Enger_Filter"
DESKTOP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_XLSX = DESKTOP_DIR / "Nutzercluster_Stichprobe.xlsx"
PCA_PNG = DESKTOP_DIR / "Nutzercluster_Stichprobe_PCA.png"
PROFILE_PNG = DESKTOP_DIR / "Nutzercluster_Stichprobe_Profile.png"
KEY_PNG = DESKTOP_DIR / "Nutzercluster_Stichprobe_Kernvariablen.png"

OUTPUT_DIR = BASE_DIR / "kundenunterschiede_outputs"
USER_FEATURES_FILE = OUTPUT_DIR / "kundenunterschiede_user_features.xlsx"
CORRELATION_MATRIX_FILE = OUTPUT_DIR / "kundenunterschiede_korrelationsmatrix.xlsx"

SAMPLE_SIZE = 2000
K_CLUSTERS = 4
RANDOM_SEED = 42
TOPICS = ["FOOD", "DRINKS", "SERVICE", "SPEED", "HYGIENE", "VALUE", "AMBIENCE", "PARKING", "ACCESSIBILITY"]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 11
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

def load_users():
    users = pd.read_excel(USER_FEATURES_FILE)
    feature_cols = [
        "full_review_count_without_chickfila",
        "full_avg_rating_without_chickfila",
        "full_rating_std_without_chickfila",
        "full_avg_review_word_count",
        "full_unique_category_count",
        "full_fast_food_category_share",
        "full_avg_price_level",
        "cfa_visited_weekend",
        "cfa_visited_morning",
        "cfa_visited_afternoon",
        "cfa_visited_evening",
        "cfa_visited_night",
        "reviewed_chickfila_more_than_once",
    ]
    feature_cols = [col for col in feature_cols if col in users.columns]
    return users, feature_cols

def transform_features(df, feature_cols, medians=None, means=None, stds=None):
    xdf = df[feature_cols].copy()
    for col in xdf.columns:
        xdf[col] = pd.to_numeric(xdf[col], errors="coerce")
    if medians is None:
        medians = xdf.median(numeric_only=True).fillna(0)
    xdf = xdf.fillna(medians)
    for col in ["full_review_count_without_chickfila"]:
        if col in xdf.columns:
            xdf[col] = np.log1p(xdf[col])
    if means is None:
        means = xdf.mean()
    if stds is None:
        stds = xdf.std(ddof=0).replace(0, 1)
    x = ((xdf - means) / stds).to_numpy(dtype=float)
    return xdf, x, medians, means, stds

def prepare_cluster_data(users, feature_cols):
    sample = users.sample(min(SAMPLE_SIZE, len(users)), random_state=RANDOM_SEED).copy()
    sample_xdf, sample_x, medians, means, stds = transform_features(sample, feature_cols)
    all_xdf, all_x, _, _, _ = transform_features(users, feature_cols, medians=medians, means=means, stds=stds)
    return sample, sample_xdf, sample_x, all_xdf, all_x, medians, means, stds

def run_kmeans(x, k=4, seed=42, max_iter=120):
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), size=k, replace=False)].copy()
    for _ in range(max_iter):
        distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        new_centers = centers.copy()
        for cluster_id in range(k):
            members = x[labels == cluster_id]
            if len(members):
                new_centers[cluster_id] = members.mean(axis=0)
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1), centers

def assign_kmeans(x, centers):
    distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = distances.argmin(axis=1)
    min_distances = np.sqrt(distances.min(axis=1))
    return labels, min_distances

def pca_2d(x):
    centered = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T

def label_cluster(row):
    avg_without_cfa = row.get("full_avg_rating_without_chickfila", row.get("avg_rating_without_chickfila", np.nan))
    if avg_without_cfa < 3:
        base = "generell kritisch"
    elif avg_without_cfa >= 4.2:
        base = "generell positiv"
    else:
        base = "mittleres Bewertungsniveau"
    if row.get("full_avg_review_word_count", row.get("avg_review_word_count", np.nan)) >= 80:
        return base + ", ausführlich"
    if row.get("full_review_count_without_chickfila", row.get("review_count_dataset", 0)) >= 20:
        return base + ", sehr aktiv"
    return base + ", gelegentlich"

def build_profiles(sample, feature_cols):
    profile_cols = [col for col in feature_cols if col in sample.columns]
    profile = sample.groupby("cluster")[profile_cols].mean()
    profile.insert(0, "user_count", sample.groupby("cluster").size())
    profile = profile.reset_index()
    profile["cluster_label"] = profile.apply(label_cluster, axis=1)
    return profile

def build_sentiment_profile(users_out):
    sentiment_cols = [f"{topic}_sentiment_mean" for topic in TOPICS if f"{topic}_sentiment_mean" in users_out.columns]
    profile = users_out.groupby("cluster")[sentiment_cols].mean().reset_index()
    profile = profile.rename(columns={f"{topic}_sentiment_mean": topic for topic in TOPICS})
    counts = users_out.groupby("cluster")[sentiment_cols].count().reset_index()
    counts = counts.rename(columns={f"{topic}_sentiment_mean": f"{topic}_n" for topic in TOPICS})
    return profile, counts

def build_cluster_description(profile, sentiment_profile):
    merged = profile.merge(sentiment_profile, on="cluster", how="left")
    rows = []
    for _, row in merged.sort_values("cluster").iterrows():
        strengths = []
        if row.get("full_review_count_without_chickfila", 0) >= merged["full_review_count_without_chickfila"].quantile(0.75):
            strengths.append("sehr aktive und erfahrene Reviewer")
        if row.get("cfa_visited_morning", 0) >= 0.8:
            strengths.append("Chick-fil-A vor allem morgens")
        if row.get("cfa_visited_afternoon", 0) >= 0.8:
            strengths.append("Chick-fil-A vor allem nachmittags")
        if row.get("cfa_visited_evening", 0) >= 0.35:
            strengths.append("Chick-fil-A haeufig abends")
        if row.get("reviewed_chickfila_more_than_once", 0) >= 0.2:
            strengths.append("haeufiger Mehrfachbewertungen")

        topic_values = {topic: row.get(topic, np.nan) for topic in TOPICS if topic in row.index}
        valid_topics = {topic: value for topic, value in topic_values.items() if pd.notna(value)}
        top_topics = sorted(valid_topics.items(), key=lambda item: item[1], reverse=True)[:3]
        low_topics = sorted(valid_topics.items(), key=lambda item: item[1])[:3]
        rows.append({
            "cluster": int(row["cluster"]),
            "user_count": int(row["user_count"]),
            "kurzbeschreibung": "; ".join(strengths) if strengths else "durchschnittliches Nutzerprofil",
            "sentiment_tendenz": f"staerker: {', '.join(f'{k} {v:.2f}' for k, v in top_topics)}; schwaecher: {', '.join(f'{k} {v:.2f}' for k, v in low_topics)}",
            "reviews_ohne_chickfila": row.get("full_review_count_without_chickfila", np.nan),
            "rating_ohne_chickfila": row.get("full_avg_rating_without_chickfila", np.nan),
            "reviewlaenge": row.get("full_avg_review_word_count", np.nan),
            "fast_food_anteil": row.get("full_fast_food_category_share", np.nan),
            "cfa_wochenende": row.get("cfa_visited_weekend", np.nan),
            "cfa_morgens": row.get("cfa_visited_morning", np.nan),
            "cfa_nachmittags": row.get("cfa_visited_afternoon", np.nan),
            "cfa_abends": row.get("cfa_visited_evening", np.nan),
            "cfa_nachts": row.get("cfa_visited_night", np.nan),
            "cfa_mehrfach": row.get("reviewed_chickfila_more_than_once", np.nan),
        })
    return pd.DataFrame(rows)

def style_plot(ax):
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")
    ax.grid(False)
    ax.tick_params(colors="black")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_color("black")

def plot_pca(coords, labels):
    fig, ax = plt.subplots(figsize=(8.5, 6))
    colors = ["#b71c1c", "#f29a9a", "#222222", "#7f0000"]
    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        ax.scatter(coords[mask, 0], coords[mask, 1], s=22, color=colors[cluster_id % len(colors)], edgecolor="black", linewidth=0.25, alpha=0.72, label=f"Cluster {cluster_id}")
    ax.set_title("Nutzercluster, PCA-Ansicht")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    style_plot(ax)
    legend = ax.legend(frameon=True, facecolor="white", edgecolor="black")
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(PCA_PNG, dpi=250)
    plt.close(fig)

def plot_profiles(profile):
    plot_cols = [
        "full_avg_rating_without_chickfila",
        "full_review_count_without_chickfila",
        "full_avg_review_word_count",
        "full_fast_food_category_share",
        "cfa_visited_weekend",
        "reviewed_chickfila_more_than_once",
    ]
    plot_cols = [col for col in plot_cols if col in profile.columns]
    data = profile.set_index("cluster")[plot_cols]
    scaled = (data - data.min()) / (data.max() - data.min()).replace(0, 1)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(plot_cols))
    width = 0.18
    colors = ["#b71c1c", "#f29a9a", "#222222", "#7f0000"]
    for idx, (cluster_id, row) in enumerate(scaled.iterrows()):
        ax.bar(x + (idx - (len(scaled) - 1) / 2) * width, row.values, width=width, label=f"Cluster {cluster_id}", color=colors[idx % len(colors)], edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_cols, rotation=30, ha="right")
    ax.set_ylabel("relativ skaliert")
    ax.set_title("Profile der Nutzercluster")
    style_plot(ax)
    legend = ax.legend(frameon=True, facecolor="white", edgecolor="black")
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(PROFILE_PNG, dpi=250)
    plt.close(fig)

def plot_key_profiles(profile):
    plot_specs = [
        ("full_avg_rating_without_chickfila", "Allgemeines Rating ohne Chick-fil-A"),
        ("full_avg_review_word_count", "Reviewlänge"),
        ("full_review_count_without_chickfila", "Reviews ohne Chick-fil-A"),
        ("full_fast_food_category_share", "Fast-Food-Anteil"),
        ("cfa_visited_weekend", "CFA am Wochenende"),
    ]
    plot_specs = [(col, label) for col, label in plot_specs if col in profile.columns]
    raw = profile.set_index("cluster")[[col for col, _ in plot_specs]].sort_index()
    scaled = (raw - raw.min()) / (raw.max() - raw.min()).replace(0, 1)
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    x = np.arange(len(scaled.index))
    width = 0.15
    colors = ["#111111", "#b71c1c", "#f29a9a", "#7f0000"]
    for idx, (col, label) in enumerate(plot_specs):
        offset = (idx - (len(plot_specs) - 1) / 2) * width
        ax.bar(x + offset, scaled[col], width=width, label=label, color=colors[idx % len(colors)], edgecolor="black", linewidth=0.8)
        for xpos, scaled_value, raw_value in zip(x + offset, scaled[col], raw[col]):
            ax.text(xpos, scaled_value + 0.02, f"{raw_value:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold", rotation=90)
    ax.set_title("Zentrale Eigenschaften nach Nutzercluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("relativ skaliert je Variable")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {int(cluster)}" for cluster in scaled.index])
    ax.set_ylim(0, 1.18)
    style_plot(ax)
    legend = ax.legend(frameon=True, facecolor="white", edgecolor="black", loc="center left", bbox_to_anchor=(1.01, 0.5))
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(KEY_PNG, dpi=250)
    plt.close(fig)

def main():
    users, feature_cols = load_users()
    sample, sample_xdf, sample_x, all_xdf, all_x, medians, means, stds = prepare_cluster_data(users, feature_cols)
    train_labels, centers = run_kmeans(sample_x, k=K_CLUSTERS, seed=RANDOM_SEED)
    all_labels, all_distances = assign_kmeans(all_x, centers)
    coords = pca_2d(sample_x)
    sample["cluster"] = train_labels
    sample["pca_1"] = coords[:, 0]
    sample["pca_2"] = coords[:, 1]
    users_out = users.copy()
    users_out["cluster"] = all_labels
    users_out["cluster_distance"] = all_distances
    users_out["is_cluster_training_sample"] = users_out["global_user_id"].isin(sample["global_user_id"]).astype(int)
    profile = build_profiles(users_out, feature_cols)
    sentiment_profile, sentiment_counts = build_sentiment_profile(users_out)
    cluster_description = build_cluster_description(profile, sentiment_profile)
    features_used = pd.DataFrame({"feature": feature_cols})
    scaling = pd.DataFrame({
        "feature": feature_cols,
        "imputation_median": medians.reindex(feature_cols).to_numpy(),
        "scaling_mean_after_imputation": means.reindex(feature_cols).to_numpy(),
        "scaling_std_after_imputation": stds.reindex(feature_cols).to_numpy(),
    })
    summary = pd.DataFrame([
        {"kennzahl": "user_gesamt", "wert": len(users)},
        {"kennzahl": "stichprobe_user", "wert": len(sample)},
        {"kennzahl": "cluster_zugeordnet_user", "wert": len(users_out)},
        {"kennzahl": "cluster", "wert": K_CLUSTERS},
        {"kennzahl": "cluster_features", "wert": len(feature_cols)},
        {"kennzahl": "avg_star_rating_als_clusterfeature", "wert": 0},
        {"kennzahl": "sentiments_als_clusterfeature", "wert": 0},
        {"kennzahl": "clustertraining", "wert": "KMeans-Zentren auf Stichprobe, Zuordnung auf alle User"},
    ])
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        users_out.to_excel(writer, sheet_name="cluster_user", index=False)
        sample.to_excel(writer, sheet_name="training_sample", index=False)
        cluster_description.to_excel(writer, sheet_name="cluster_beschreibung", index=False)
        profile.to_excel(writer, sheet_name="cluster_profile", index=False)
        sentiment_profile.to_excel(writer, sheet_name="sentiment_profile", index=False)
        sentiment_counts.to_excel(writer, sheet_name="sentiment_counts", index=False)
        features_used.to_excel(writer, sheet_name="features_used", index=False)
        scaling.to_excel(writer, sheet_name="imputation_scaling", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
    plot_pca(coords, train_labels)
    plot_profiles(profile)
    plot_key_profiles(profile)
    print(f"Excel: {OUTPUT_XLSX}")
    print(f"PCA: {PCA_PNG}")
    print(f"Profile: {PROFILE_PNG}")
    print(f"Kernvariablen: {KEY_PNG}")

if __name__ == "__main__":
    main()

