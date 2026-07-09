from pathlib import Path
import os
import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DESKTOP_DIR = Path.home() / "Desktop" / "ChickFilA_Diagramme_Enger_Filter"
USER_BIAS_FILE = BASE_DIR / "kundenunterschiede_outputs" / "user_bias_full_datasets_ohne_chickfila_min_5.xlsx"

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

if __name__ == "__main__":
    userrating_verteilung()

