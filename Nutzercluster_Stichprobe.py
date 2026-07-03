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

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 11
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

def load_users():
    users = pd.read_excel(USER_FEATURES_FILE)
    corr = pd.read_excel(CORRELATION_MATRIX_FILE, index_col=0)
    feature_cols = [col for col in corr.columns if col in users.columns and col != "avg_star_rating"]
    return users, feature_cols

def prepare_cluster_data(users, feature_cols):
    sample = users.sample(min(SAMPLE_SIZE, len(users)), random_state=RANDOM_SEED).copy()
    xdf = sample[feature_cols].copy()
    for col in xdf.columns:
        xdf[col] = pd.to_numeric(xdf[col], errors="coerce")
        xdf[col] = xdf[col].fillna(xdf[col].median())
    for col in ["review_count_dataset", "total_review_words", "yelp_user_review_count"]:
        if col in xdf.columns:
            xdf[col] = np.log1p(xdf[col])
    means = xdf.mean()
    stds = xdf.std(ddof=0).replace(0, 1)
    x = ((xdf - means) / stds).to_numpy(dtype=float)
    return sample, xdf, x

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

def pca_2d(x):
    centered = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T

def label_cluster(row):
    if row.get("avg_rating_without_chickfila", np.nan) < 3:
        base = "generell kritisch"
    elif row.get("avg_rating_without_chickfila", np.nan) >= 4.2:
        base = "generell positiv"
    else:
        base = "mittleres Bewertungsniveau"
    if row.get("avg_review_word_count", np.nan) >= 80:
        return base + ", ausführlich"
    if row.get("review_count_dataset", 0) >= 2:
        return base + ", mehrfach aktiv"
    return base + ", gelegentlich"

def build_profiles(sample, feature_cols):
    profile_cols = [col for col in feature_cols if col in sample.columns]
    profile = sample.groupby("cluster")[profile_cols].mean()
    profile.insert(0, "user_count", sample.groupby("cluster").size())
    profile = profile.reset_index()
    profile["cluster_label"] = profile.apply(label_cluster, axis=1)
    return profile

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
    plot_cols = ["avg_rating_without_chickfila", "review_count_dataset", "avg_review_word_count", "weekend_share", "avg_local_hour", "FOOD_sentiment_mean", "SERVICE_sentiment_mean", "VALUE_sentiment_mean"]
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
        ("avg_rating_without_chickfila", "Allgemeines Rating ohne Chick-fil-A"),
        ("avg_review_word_count", "Reviewlänge"),
        ("review_count_dataset", "Chick-fil-A-Reviews je User"),
        ("weekend_share", "Wochenendanteil"),
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
    sample, xdf, x = prepare_cluster_data(users, feature_cols)
    labels, centers = run_kmeans(x, k=K_CLUSTERS, seed=RANDOM_SEED)
    coords = pca_2d(x)
    sample["cluster"] = labels
    sample["pca_1"] = coords[:, 0]
    sample["pca_2"] = coords[:, 1]
    profile = build_profiles(sample, feature_cols)
    features_used = pd.DataFrame({"feature": feature_cols})
    summary = pd.DataFrame([
        {"kennzahl": "user_gesamt", "wert": len(users)},
        {"kennzahl": "stichprobe_user", "wert": len(sample)},
        {"kennzahl": "cluster", "wert": K_CLUSTERS},
        {"kennzahl": "cluster_features", "wert": len(feature_cols)},
        {"kennzahl": "avg_star_rating_als_clusterfeature", "wert": 0},
    ])
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        sample.to_excel(writer, sheet_name="cluster_user", index=False)
        profile.to_excel(writer, sheet_name="cluster_profile", index=False)
        features_used.to_excel(writer, sheet_name="features_used", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
    plot_pca(coords, labels)
    plot_profiles(profile)
    plot_key_profiles(profile)
    print(f"Excel: {OUTPUT_XLSX}")
    print(f"PCA: {PCA_PNG}")
    print(f"Profile: {PROFILE_PNG}")
    print(f"Kernvariablen: {KEY_PNG}")

if __name__ == "__main__":
    main()
