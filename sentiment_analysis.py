"""
Focus Group Emotional Sentiment Analysis
=========================================
Analyzes a qualitative transcript using VADER, TextBlob, and NRC emotion lexicon.
Outputs a pandas DataFrame with segment-level and summary-level emotional profiles.

Required packages:  pip install vaderSentiment textblob nrclex nltk pandas
"""

import re
import sys
import io
import warnings
import pandas as pd
import nltk

# Force UTF-8 output so box-drawing and block characters render correctly on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from nrclex import NRCLex

warnings.filterwarnings("ignore")

# Download required NLTK resources (silent after first run)
for resource in ("punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger",
                 "averaged_perceptron_tagger_eng", "wordnet", "omw-1.4"):
    nltk.download(resource, quiet=True)

from nltk.corpus import stopwords

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
FILE_PATH = r"c:\Users\never\OneDrive\Desktop\D&D Sentiment Analysis\5726 - DD focus group recording (1).txt"
OUTPUT_CSV = r"c:\Users\never\OneDrive\Desktop\D&D Sentiment Analysis\sentiment_analysis_output.csv"
MIN_WORDS = 15   # Skip fragments shorter than this (backchannels, crosstalk, etc.)

# ─── STOP WORD LIST ───────────────────────────────────────────────────────────
# Standard English stop words + transcript-specific filler language
STOP_WORDS = set(stopwords.words("english"))
TRANSCRIPT_FILLER = {
    "like", "know", "think", "yeah", "okay", "just", "really", "kind", "sort",
    "thing", "things", "well", "maybe", "also", "going", "got", "get", "bit",
    "little", "lot", "sometimes", "even", "always", "never", "still", "already",
    "now", "say", "said", "way", "something", "someone", "people", "person",
    "want", "feel", "feeling", "feels", "felt", "make", "makes", "made", "come",
    "comes", "came", "go", "goes", "went", "one", "two", "three", "much", "many",
    "every", "sure", "totally", "actually", "probably", "definitely", "honestly",
    "basically", "especially", "quite", "super", "pretty", "right", "mean",
    "dont", "im", "ive", "thats", "youre", "theyre", "could", "would", "should",
    "might", "need", "try", "think", "thought", "see", "seen", "take", "taking",
    "put", "tell", "told", "ask", "asked", "around", "back", "really", "because",
}
STOP_WORDS.update(TRANSCRIPT_FILLER)

# ─── NLP TOOLS ────────────────────────────────────────────────────────────────
vader = SentimentIntensityAnalyzer()

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def parse_transcript(filepath):
    """Split transcript into speaking turns (paragraphs separated by blank lines)."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    # Skip the first line (date/title header)
    paragraphs = re.split(r"\n\s*\n", raw.strip())
    turns = []
    for p in paragraphs[1:]:          # skip header
        text = " ".join(p.split())    # collapse internal newlines / extra spaces
        if len(text.split()) >= MIN_WORDS:
            turns.append(text)
    return turns


def clean_keywords(text, n=12):
    """Return top n meaningful content words after stop word removal."""
    tokens = word_tokenize(text.lower())
    words = [t for t in tokens if t.isalpha() and len(t) > 2 and t not in STOP_WORDS]
    # Preserve order, deduplicate while keeping first occurrence
    seen, unique = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return ", ".join(unique[:n])


def nrc_profile(text):
    """Return a dict of NRC emotion proportions (Plutchik's 8 basic emotions)."""
    emotion_obj = NRCLex("placeholder")
    emotion_obj.load_raw_text(text)
    freq = emotion_obj.affect_frequencies
    # Keep only the 8 basic emotions (exclude the generic positive/negative keys)
    basic = {k: freq.get(k, 0.0) for k in
             ("joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation")}
    total = sum(basic.values()) or 1.0
    return {k: round(v / total, 3) for k, v in basic.items()}


def dominant_emotion(profile):
    """Return the highest-scoring NRC emotion, or 'Neutral' if all are zero."""
    if not any(profile.values()):
        return "Neutral"
    return max(profile, key=profile.get).capitalize()


def valence_label(compound):
    """Translate VADER compound score to a plain-language valence label."""
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def intensity_label(compound):
    """Qualitative intensity based on absolute compound score."""
    abs_c = abs(compound)
    if abs_c >= 0.6:
        return "Strong"
    if abs_c >= 0.3:
        return "Moderate"
    if abs_c >= 0.05:
        return "Mild"
    return "Neutral"


# ─── MAIN ANALYSIS ────────────────────────────────────────────────────────────

def analyze_transcript(filepath):
    turns = parse_transcript(filepath)

    records = []
    for i, turn in enumerate(turns, start=1):
        # VADER
        vs = vader.polarity_scores(turn)
        compound = vs["compound"]

        # TextBlob
        blob = TextBlob(turn)
        polarity = round(blob.sentiment.polarity, 3)
        subjectivity = round(blob.sentiment.subjectivity, 3)

        # NRC
        nrc = nrc_profile(turn)
        dom = dominant_emotion(nrc)

        # Excerpt for readability (first 25 words)
        words = turn.split()
        excerpt = " ".join(words[:25]) + ("..." if len(words) > 25 else "")

        records.append({
            # ── Identification ──
            "Segment_Number":           i,
            "Word_Count":               len(words),
            "Excerpt":                  excerpt,

            # ── VADER Sentiment ──
            # Compound: -1.0 (most negative) → +1.0 (most positive); |score| > 0.05 = non-neutral
            "VADER_Compound_Score":     round(compound, 3),
            "VADER_Positive_Content":   round(vs["pos"], 3),
            "VADER_Negative_Content":   round(vs["neg"], 3),
            "VADER_Neutral_Content":    round(vs["neu"], 3),
            "Overall_Emotional_Valence": valence_label(compound),
            "Emotional_Intensity":      intensity_label(compound),

            # ── TextBlob ──
            # Polarity: -1 (negative) → +1 (positive)
            # Subjectivity: 0 (factual/objective) → 1 (highly personal/emotional)
            "Polarity_Score":           polarity,
            "Subjectivity_Score":       subjectivity,

            # ── NRC Plutchik Emotion Wheel (proportions, sum ≈ 1.0) ──
            "Dominant_Emotion":         dom,
            "Joy":                      nrc["joy"],
            "Trust":                    nrc["trust"],
            "Fear":                     nrc["fear"],
            "Surprise":                 nrc["surprise"],
            "Sadness":                  nrc["sadness"],
            "Disgust":                  nrc["disgust"],
            "Anger":                    nrc["anger"],
            "Anticipation":             nrc["anticipation"],

            # ── Lexical Content ──
            "Key_Emotional_Words":      clean_keywords(turn),
        })

    return pd.DataFrame(records)


# ─── PRINT CLINICAL SUMMARY ───────────────────────────────────────────────────

def print_summary(df):
    sep = "=" * 72
    print(sep)
    print("  FOCUS GROUP EMOTIONAL SENTIMENT ANALYSIS")
    print("  Source: 5726 – D&D Focus Group Recording")
    print(sep)

    n = len(df)
    print(f"\n  Segments analyzed : {n}")
    print(f"  Total words       : {df['Word_Count'].sum():,}")
    print(f"  Mean segment length: {df['Word_Count'].mean():.0f} words")

    # ── Valence distribution ──
    print("\n" + "─" * 72)
    print("  OVERALL EMOTIONAL VALENCE  (VADER compound score)")
    print("  Note: Compound ranges from -1.0 (most negative) to +1.0 (most positive).")
    print("  Segments with |compound| < 0.05 are classified as Neutral.")
    print("─" * 72)
    vc = df["Overall_Emotional_Valence"].value_counts()
    for label in ["Positive", "Neutral", "Negative"]:
        count = vc.get(label, 0)
        pct = count / n * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:<10}: {count:>3} segments  ({pct:4.1f}%)  {bar}")

    mean_c = df["VADER_Compound_Score"].mean()
    mean_pos = df["VADER_Positive_Content"].mean()
    mean_neg = df["VADER_Negative_Content"].mean()
    print(f"\n  Mean Compound Score    : {mean_c:+.3f}")
    print(f"  Mean Positive Content  : {mean_pos:.3f}")
    print(f"  Mean Negative Content  : {mean_neg:.3f}")

    # ── Intensity distribution ──
    print("\n" + "─" * 72)
    print("  EMOTIONAL INTENSITY DISTRIBUTION")
    print("─" * 72)
    ic = df["Emotional_Intensity"].value_counts()
    for label in ["Strong", "Moderate", "Mild", "Neutral"]:
        count = ic.get(label, 0)
        pct = count / n * 100
        print(f"  {label:<10}: {count:>3} segments  ({pct:4.1f}%)")

    # ── TextBlob ──
    print("\n" + "─" * 72)
    print("  LANGUAGE CHARACTERISTICS  (TextBlob)")
    print("  Polarity:    -1.0 = strongly negative  →  +1.0 = strongly positive")
    print("  Subjectivity: 0.0 = objective/factual  →   1.0 = highly personal/emotional")
    print("─" * 72)
    print(f"  Mean Polarity      : {df['Polarity_Score'].mean():+.3f}")
    print(f"  Mean Subjectivity  : {df['Subjectivity_Score'].mean():.3f}")

    # ── NRC emotion profile ──
    print("\n" + "─" * 72)
    print("  NRC PLUTCHIK EMOTION PROFILE  (mean proportional presence)")
    print("  Based on Plutchik's (1980) Wheel of Emotions.")
    print("  Higher values = that emotion more prevalent across the transcript.")
    print("─" * 72)
    nrc_cols = ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation"]
    nrc_means = df[nrc_cols].mean().sort_values(ascending=False)
    for emotion, val in nrc_means.items():
        bar = "█" * int(val * 50)
        print(f"  {emotion:<14}: {val:.3f}  {bar}")

    print("\n  Dominant emotion per segment:")
    dc = df["Dominant_Emotion"].value_counts()
    for emotion, count in dc.items():
        pct = count / n * 100
        print(f"    {emotion:<14}: {count:>3} segments  ({pct:4.1f}%)")

    # ── Peak segments ──
    print("\n" + "─" * 72)
    print("  PEAK EMOTIONAL SEGMENTS")
    print("─" * 72)

    print("\n  Three most POSITIVE segments (highest VADER compound score):")
    top_pos = df.nlargest(3, "VADER_Compound_Score")[
        ["Segment_Number", "VADER_Compound_Score", "Dominant_Emotion", "Excerpt"]
    ]
    for _, row in top_pos.iterrows():
        print(f"    Seg {int(row['Segment_Number']):>3}  score={row['VADER_Compound_Score']:+.3f}"
              f"  [{row['Dominant_Emotion']}]  \"{row['Excerpt'][:70]}\"")

    print("\n  Three most NEGATIVE segments (lowest VADER compound score):")
    top_neg = df.nsmallest(3, "VADER_Compound_Score")[
        ["Segment_Number", "VADER_Compound_Score", "Dominant_Emotion", "Excerpt"]
    ]
    for _, row in top_neg.iterrows():
        print(f"    Seg {int(row['Segment_Number']):>3}  score={row['VADER_Compound_Score']:+.3f}"
              f"  [{row['Dominant_Emotion']}]  \"{row['Excerpt'][:70]}\"")

    print("\n  Three most EMOTIONALLY INTENSE segments (highest |compound|):")
    df["_abs"] = df["VADER_Compound_Score"].abs()
    top_int = df.nlargest(3, "_abs")[
        ["Segment_Number", "VADER_Compound_Score", "Dominant_Emotion", "Excerpt"]
    ]
    df.drop(columns=["_abs"], inplace=True)
    for _, row in top_int.iterrows():
        print(f"    Seg {int(row['Segment_Number']):>3}  score={row['VADER_Compound_Score']:+.3f}"
              f"  [{row['Dominant_Emotion']}]  \"{row['Excerpt'][:70]}\"")

    print("\n" + "─" * 72)
    print("  EMOTIONAL ARC  (VADER compound score across segments)")
    print("─" * 72)
    # Simple ASCII sparkline of compound score trajectory
    scores = df["VADER_Compound_Score"].tolist()
    buckets = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    sparkline = ""
    for s in scores:
        idx = int((s + 1) / 2 * 8)   # map [-1, 1] → [0, 8]
        idx = max(0, min(8, idx))
        sparkline += buckets[idx]
    print(f"\n  Negative ←  [{sparkline}]  → Positive")
    print(f"  (each character = one transcript segment, left to right)")

    print("\n" + sep)
    print("  DATA GLOSSARY  (for non-coders)")
    print(sep)
    glossary = [
        ("Segment_Number",           "Sequential order of speaking turns in the transcript"),
        ("Word_Count",               "Number of words in this speaking turn"),
        ("Excerpt",                  "First 25 words of the speaking turn"),
        ("VADER_Compound_Score",     "Overall sentiment: –1.0 (very negative) to +1.0 (very positive)"),
        ("VADER_Positive_Content",   "Proportion of words flagged as positive (0–1)"),
        ("VADER_Negative_Content",   "Proportion of words flagged as negative (0–1)"),
        ("VADER_Neutral_Content",    "Proportion of words flagged as neutral (0–1)"),
        ("Overall_Emotional_Valence","Simplified label: Positive / Neutral / Negative"),
        ("Emotional_Intensity",      "Strength of sentiment: Strong / Moderate / Mild / Neutral"),
        ("Polarity_Score",           "TextBlob polarity: –1.0 (negative) to +1.0 (positive)"),
        ("Subjectivity_Score",       "TextBlob subjectivity: 0 (factual) to 1.0 (emotional/personal)"),
        ("Dominant_Emotion",         "The NRC emotion category most present in this segment"),
        ("Joy … Anticipation",       "NRC Plutchik emotion proportions (0–1); sum ≈ 1.0 per segment"),
        ("Key_Emotional_Words",      "Content words remaining after stop-word removal"),
    ]
    for col, desc in glossary:
        print(f"  {col:<30}  {desc}")

    print()


# ─── RUN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = analyze_transcript(FILE_PATH)
    print_summary(df)

    # Display full DataFrame
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 55)
    pd.set_option("display.float_format", "{:.3f}".format)
    print("\n\nFULL SEGMENT-BY-SEGMENT RESULTS\n")
    print(df.to_string(index=False))

    # Save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n\nResults saved to:\n  {OUTPUT_CSV}")
