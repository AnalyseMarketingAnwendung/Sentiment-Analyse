from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd
from wordcloud import WordCloud


BASE = Path("/Users/clara/PycharmProjects/pythonProject5")
PROJECT4 = Path("/Users/clara/PycharmProjects/pythonProject4")
OUT = BASE / "analysis" / "chickfila_review_wordcloud"
YELP_FINAL = PROJECT4 / "data" / "yelp_final.csv"
CHICK_CLEAN = PROJECT4 / "data" / "ChickFilA_Bereinigt.xlsx"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


STOPWORDS = {
    "a", "about", "after", "again", "all", "also", "am", "an", "and", "any", "are", "as",
    "at", "back", "be", "because", "been", "before", "being", "but", "by", "can", "could",
    "did", "didnt", "do", "does", "dont", "for", "from", "get", "go", "going", "got",
    "had", "has", "have", "he", "her", "here", "him", "his", "how", "i", "if", "im",
    "in", "into", "is", "it", "its", "ive", "just", "like", "me", "my", "no", "not",
    "of", "on", "one", "or", "our", "out", "over", "really", "so", "some", "than",
    "that", "the", "their", "them", "there", "they", "this", "to", "too", "up", "us",
    "was", "we", "were", "what", "when", "where", "which", "who", "will", "with",
    "would", "you", "your", "youre", "very", "more", "most", "much", "even", "still",
    "always", "never", "ordered", "order", "orders", "place", "location", "restaurant",
    "chick", "fil", "chickfil", "chickfila", "cfa", "fil-a", "fila",
}


def load_chick_clean_texts() -> pd.Series:
    df = pd.read_excel(CHICK_CLEAN, sheet_name="Alle_Daten", usecols=["text"])
    return df["text"].dropna().astype(str)


def load_yelp_chickfila_texts() -> pd.Series:
    texts = []
    usecols = ["text", "business_name"]
    for chunk in pd.read_csv(YELP_FINAL, usecols=usecols, chunksize=100_000):
        mask = chunk["business_name"].astype(str).str.contains("chick", case=False, na=False)
        if mask.any():
            texts.extend(chunk.loc[mask, "text"].dropna().astype(str).tolist())
    return pd.Series(texts, dtype=str)


def tokenize(texts: pd.Series) -> Counter:
    counter: Counter[str] = Counter()
    for text in texts:
        text = text.lower().replace("’", "'")
        text = re.sub(r"[^a-zA-Z\s']", " ", text)
        for token in re.findall(r"[a-z][a-z']{2,}", text):
            token = token.replace("'", "")
            if token not in STOPWORDS and len(token) >= 3:
                counter[token] += 1
    return counter


def draw_wordcloud(counter: Counter, output: Path) -> None:
    cloud = WordCloud(
        width=1800,
        height=900,
        background_color="white",
        max_words=320,
        prefer_horizontal=0.72,
        collocations=False,
        normalize_plurals=True,
        relative_scaling=0.45,
        min_font_size=8,
        max_font_size=180,
        margin=3,
        random_state=7,
        font_path=FONT_BOLD,
        colormap="tab10",
    ).generate_from_frequencies(dict(counter))
    cloud.to_file(str(output))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    chick_texts = load_chick_clean_texts()
    yelp_texts = load_yelp_chickfila_texts()
    all_texts = pd.concat([chick_texts, yelp_texts], ignore_index=True)
    counter = tokenize(all_texts)

    top_words = pd.DataFrame(counter.most_common(250), columns=["word", "count"])
    top_words.to_csv(OUT / "chickfila_review_word_frequencies.csv", index=False)
    summary = pd.DataFrame(
        [
            ["chickfila_bereinigt_reviews", len(chick_texts)],
            ["yelp_final_chickfila_reviews", len(yelp_texts)],
            ["combined_reviews", len(all_texts)],
            ["unique_words_after_cleaning", len(counter)],
        ],
        columns=["metric", "value"],
    )
    summary.to_csv(OUT / "summary.csv", index=False)
    draw_wordcloud(counter, OUT / "chickfila_review_wordcloud.png")
    draw_wordcloud(counter, OUT / "chickfila_review_wordcloud_more_words.png")


if __name__ == "__main__":
    main()

