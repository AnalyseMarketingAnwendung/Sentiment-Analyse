from pathlib import Path
import os
import re
import textwrap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DESKTOP_DIR = Path.home() / "Desktop" / "ChickFilA_Diagramme_Enger_Filter"
USER_FEATURES_FILE = BASE_DIR / "kundenunterschiede_outputs" / "kundenunterschiede_user_features.xlsx"
TOPIC_COLS = ["FOOD", "DRINKS", "SERVICE", "SPEED", "HYGIENE", "VALUE", "AMBIENCE", "PARKING", "ACCESSIBILITY"]

def safe_sheet_name(name):
    return re.sub(r"[\[\]:*?/\\]", "_", name)[:31]

def save_named_excel(name, sheets):
    path = DESKTOP_DIR / f"{name}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, df in sheets.items():
            df.to_excel(writer, sheet_name=safe_sheet_name(sheet), index=False)
    return path

def plot_heatmap(matrix, name, title, cbar_label, vmin=-1, vmax=1, legend_text=None):
    values = matrix.to_numpy(dtype=float)
    if vmin is None or vmax is None:
        vmax_abs = np.nanmax(np.abs(values))
        vmin, vmax = -vmax_abs, vmax_abs
    fig, ax = plt.subplots(figsize=(17 if legend_text else 13.5, 7))
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
    if legend_text:
        wrapped = "\n".join(textwrap.fill(line, width=46) for line in legend_text.splitlines())
        fig.text(
            0.765,
            0.5,
            wrapped,
            ha="left",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "square,pad=0.7"},
        )
        fig.tight_layout(rect=[0, 0, 0.74, 1])
    else:
        fig.tight_layout()
    fig.savefig(DESKTOP_DIR / f"{name}.png", dpi=250)
    plt.close(fig)

def kundenmatrix():
    features = pd.read_excel(USER_FEATURES_FILE)
    row_vars = [
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
    col_vars = ["cfa_avg_star_rating"] + [f"{topic}_sentiment_mean" for topic in TOPIC_COLS]
    labels = {
        "full_review_count_without_chickfila": "Anzahl der geschriebenen Reviews auf Yelp/Google",
        "full_avg_rating_without_chickfila": "Ø Rating ohne Chick-fil-A",
        "full_rating_std_without_chickfila": "Rating-Streuung ohne Chick-fil-A",
        "full_avg_review_word_count": "Ø Reviewlänge",
        "full_unique_category_count": "Anzahl bewertete Kategorien",
        "full_fast_food_category_share": "Anteil Fast-Food-Kategorien",
        "full_avg_price_level": "Durchschnittliches Preisniveau besuchter Restaurants",
        "cfa_visited_weekend": "Hat Filiale am Wochenende besucht",
        "cfa_visited_morning": "Hat Filiale morgens besucht",
        "cfa_visited_afternoon": "Hat Filiale nachmittags besucht",
        "cfa_visited_evening": "Hat Filiale abends besucht",
        "cfa_visited_night": "Hat Filiale nachts besucht",
        "reviewed_chickfila_more_than_once": "Chick-fil-A mehrfach bewertet",
        "cfa_avg_star_rating": "Chick-fil-A Ø Sterne",
        "FOOD_sentiment_mean": "Food",
        "DRINKS_sentiment_mean": "Drinks",
        "SERVICE_sentiment_mean": "Service",
        "SPEED_sentiment_mean": "Speed",
        "HYGIENE_sentiment_mean": "Hygiene",
        "VALUE_sentiment_mean": "Value",
        "AMBIENCE_sentiment_mean": "Ambience",
        "PARKING_sentiment_mean": "Parking",
        "ACCESSIBILITY_sentiment_mean": "Accessibility",
    }
    row_vars = [col for col in row_vars if col in features.columns]
    col_vars = [col for col in col_vars if col in features.columns]
    matrix = pd.DataFrame(index=row_vars, columns=col_vars, dtype=float)
    for row in row_vars:
        for col in col_vars:
            matrix.loc[row, col] = pd.to_numeric(features[row], errors="coerce").corr(pd.to_numeric(features[col], errors="coerce"))
    display_matrix = matrix.rename(index=labels, columns=labels)
    legend_text = "\n".join([
        "Berechnung der Kundenvariablen:",
        "Reviews: Anzahl aller Yelp/Google-Reviews des Users ohne Chick-fil-A.",
        "Ø Rating: durchschnittliches Sterne-Rating des Users ohne Chick-fil-A.",
        "Rating-Streuung: Standardabweichung dieser Ratings.",
        "Ø Reviewlänge: durchschnittliche Wortzahl der Reviews.",
        "Kategorien: Anzahl unterschiedlicher bewerteter Kategorien.",
        "Fast-Food-Anteil: Anteil realer Kategorie-Elemente wie Fast Food, Burgers, Sandwiches, Pizza, Tacos usw.",
        r"Preisniveau: Durchschnitt aus \$, \$\$, \$\$\$, \$\$\$\$ der besuchten Restaurants.",
        "Wochenende/Tageszeit: 1, wenn der User Chick-fil-A mindestens einmal zu diesem lokalen Zeitpunkt bewertet hat.",
        "Mehrfach bewertet: 1, wenn der User Chick-fil-A mehr als einmal bewertet hat.",
        "Sentiments: Durchschnitt der ABSA-Topicwerte je User.",
    ])
    plot_heatmap(
        display_matrix,
        "Kovarianzmatrix_Kundeneigenschaften",
        "Kundenverhalten im Gesamtdatensatz vs. Chick-fil-A-Rating und Sentiments",
        "Korrelation",
        legend_text=legend_text,
    )
    save_named_excel("Kovarianzmatrix_Kundeneigenschaften", {"korrelationsmatrix": display_matrix.reset_index(names="kundenbeschreibung"), "user_features": features})

if __name__ == "__main__":
    kundenmatrix()

