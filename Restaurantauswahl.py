# vorher installieren:
# pip install pandas openpyxl

import pandas as pd

# Excel laden
df = pd.read_excel(
    "../data/ChickFilA_Bereinigt.xlsx"
)

# Spalten bereinigen
df.columns = df.columns.str.strip().str.lower()

# Kategorien filtern
allowed_categories = [
    "american restaurant", "barbecue restaurant", "breakfast restaurant", "brunch restaurant",
    "buffet restaurant", "carribean restaurant", "cheesesteak restaurant", "chicken restaurant",
    "chicken wings restaurant", "contemporary louisiana", "continental restaurant", "delivery restaurant",
    "eclectic restaurant", "fast food restaurant", "family restaurant", "fine dining restaurant", "hamburger restaurant",
    "health food restaurant", "hot dog restaurant", "latin american restaurant", "lunch restaurant",
    "mexican restaurant", "mid-atlantic restaurant (us)", "new american restaurant", "nuevo latino restaurant",
    "oyster bar restaurant", "pan-latin restaurant", "peruvian restaurant", "pizza restaurant",
    "restaurant", "seafood restaurant", "soul food restaurant", "soup restaurant", "southern restaurant (us)",
    "southwestern restaurant (us)", "taco restaurant", "takeout restaurant","traditional american", "tex-mex restaurant",
    "vegan restaurant", "vegetarian restaurant"]


df = df[
    df["category"]
    .str.strip()
    .str.lower()
    .isin(allowed_categories)
]

# -------------------------------------------------
# 1) Nur Restaurants mit mehreren Filialen behalten
# -------------------------------------------------

filial_counts = (
    df.groupby("business_name")["store_id"]
    .nunique()
)

mehrere_filialen_namen = filial_counts[
    filial_counts > 1
].index

# Datensatz reduzieren
df = df[
    df["business_name"].isin(mehrere_filialen_namen)
]

# -------------------------------------------------
# 2) Durchschnittsbewertung pro Filiale berechnen
# -------------------------------------------------

filialen_level = (
    df.groupby(["business_name", "store_id"])
    .agg(
        filial_bewertung=("rating", "mean"),
        anzahl_reviews_filiale=("rating", "size"),
        category=("category", "first")
    )
    .reset_index()
)

# -------------------------------------------------
# 3) Statistik auf Restaurant-/Kettenebene
# -------------------------------------------------

restaurant_stats = (
    filialen_level.groupby("business_name")
    .agg(
        category=("category", "first"),
        anzahl_filialen=("store_id", "nunique"),
        anzahl_bewertungen=("anzahl_reviews_filiale", "sum"),
        mittlere_filialbewertung=("filial_bewertung", "mean"),
        varianz_zwischen_filialen=("filial_bewertung", "var")
    )
    .reset_index()
)

# -------------------------------------------------
# 4) Spaltenreihenfolge festlegen
# -------------------------------------------------

restaurant_stats = restaurant_stats[
    [
        "business_name",
        "category",
        "anzahl_filialen",
        "anzahl_bewertungen",
        "mittlere_filialbewertung",
        "varianz_zwischen_filialen"
    ]
]

# -------------------------------------------------
# 5) Nach Anzahl Filialen sortieren
# -------------------------------------------------

restaurant_stats = restaurant_stats.sort_values(
    by="anzahl_filialen",
    ascending=False
)

# -------------------------------------------------
# 6) Ergebnis als Excel speichern
# -------------------------------------------------

restaurant_stats.to_excel(
    "../data/filialauswahl2.xlsx",
    index=False
)
