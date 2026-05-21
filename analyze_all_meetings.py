"""
Focus Group — Multi-Meeting Sentiment Analysis & Visualization
==============================================================
Analyzes every .txt file in the Meetings/ folder and produces:
  Per-meeting  : one summary figure per meeting (valence, emotions, arc)
  Cross-meeting: three comparison figures across all meetings
All figures saved to Graphs/

Run once; re-running overwrites previous output.

Required packages: pip install vaderSentiment textblob nrclex nltk pandas seaborn matplotlib
"""

import sys, io, re, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # no display needed — just save files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from nrclex import NRCLex
import nltk
from nltk.tokenize import word_tokenize

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

for r in ("punkt", "punkt_tab", "stopwords", "wordnet",
          "omw-1.4", "averaged_perceptron_tagger_eng"):
    nltk.download(r, quiet=True)
from nltk.corpus import stopwords


# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(r"c:\Users\never\OneDrive\Desktop\D&D Sentiment Analysis")
MEETINGS_DIR = BASE_DIR / "Meetings"
GRAPHS_DIR   = BASE_DIR / "Graphs"
GRAPHS_DIR.mkdir(exist_ok=True)

# ── GLOBAL STYLE ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "sans-serif",
})

EMOTION_COLS = ["Joy", "Trust", "Fear", "Surprise",
                "Sadness", "Disgust", "Anger", "Anticipation"]

EMOTION_COLORS = {
    "Joy":          "#F4D03F",
    "Trust":        "#27AE60",
    "Fear":         "#8E44AD",
    "Surprise":     "#F39C12",
    "Sadness":      "#2980B9",
    "Disgust":      "#795548",
    "Anger":        "#E74C3C",
    "Anticipation": "#1ABC9C",
    "Neutral":      "#BDC3C7",
}

VALENCE_COLORS = {"Positive": "#27AE60", "Neutral": "#BDC3C7", "Negative": "#E74C3C"}

# ── NLP SETUP ─────────────────────────────────────────────────────────────────
vader = SentimentIntensityAnalyzer()

STOP_WORDS = set(stopwords.words("english")) | {
    "like", "know", "think", "yeah", "okay", "just", "really", "kind", "sort",
    "thing", "things", "well", "maybe", "also", "going", "got", "get", "bit",
    "little", "lot", "sometimes", "even", "always", "never", "still", "already",
    "now", "say", "said", "way", "something", "someone", "people", "person",
    "want", "feel", "feeling", "feels", "felt", "make", "makes", "made",
    "come", "comes", "came", "go", "goes", "went", "one", "two", "three",
    "much", "many", "every", "sure", "totally", "actually", "probably",
    "definitely", "honestly", "basically", "especially", "quite", "super",
    "pretty", "right", "mean", "could", "would", "should", "might", "need",
    "try", "thought", "see", "seen", "take", "taking", "put", "tell", "told",
    "ask", "asked", "around", "back", "im", "ive", "thats", "youre", "theyre",
    "dont", "doesnt", "didnt", "cant", "wont", "wasnt", "isnt", "oh", "uh",
}

MIN_WORDS = 18   # filter out backchannels and short crosstalk fragments


# ── HELPERS ───────────────────────────────────────────────────────────────────

def meeting_label(filepath: Path) -> str:
    """Turn a filename like M2_SocialGroups.txt into a clean label."""
    stem = filepath.stem                       # e.g. M2_SocialGroups
    parts = stem.split("_", 1)
    num   = parts[0]                           # M2
    topic = parts[1].replace("_", " ") if len(parts) > 1 else ""
    # Insert spaces before capital letters (CamelCase → readable)
    topic = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", topic)
    return f"{num}: {topic}"


def parse_transcript(filepath: Path) -> list[str]:
    """Split transcript into speaking turns; skip very short fragments."""
    text = filepath.read_text(encoding="utf-8", errors="replace")
    paragraphs = re.split(r"\n\s*\n", text.strip())
    turns = []
    for p in paragraphs[1:]:                  # skip title line
        clean = " ".join(p.split())
        if len(clean.split()) >= MIN_WORDS:
            turns.append(clean)
    return turns


def nrc_profile(text: str) -> dict:
    obj = NRCLex("placeholder")
    obj.load_raw_text(text)
    freq = obj.affect_frequencies
    basic = {k: freq.get(k, 0.0) for k in
             ("joy","trust","fear","surprise","sadness","disgust","anger","anticipation")}
    total = sum(basic.values()) or 1.0
    return {k.capitalize(): round(v / total, 4) for k, v in basic.items()}


def valence_label(c: float) -> str:
    if c >= 0.05:  return "Positive"
    if c <= -0.05: return "Negative"
    return "Neutral"


def intensity_label(c: float) -> str:
    a = abs(c)
    if a >= 0.6:  return "Strong"
    if a >= 0.3:  return "Moderate"
    if a >= 0.05: return "Mild"
    return "Neutral"


def analyze_file(filepath: Path) -> pd.DataFrame:
    turns = parse_transcript(filepath)
    rows  = []
    for i, turn in enumerate(turns, 1):
        vs  = vader.polarity_scores(turn)
        c   = vs["compound"]
        blb = TextBlob(turn)
        nrc = nrc_profile(turn)
        exc = " ".join(turn.split()[:20]) + ("..." if len(turn.split()) > 20 else "")
        rows.append({
            "Segment":        i,
            "Word_Count":     len(turn.split()),
            "Excerpt":        exc,
            "Compound":       round(c, 4),
            "Positive_Pct":   round(vs["pos"], 4),
            "Negative_Pct":   round(vs["neg"], 4),
            "Neutral_Pct":    round(vs["neu"], 4),
            "Valence":        valence_label(c),
            "Intensity":      intensity_label(c),
            "Polarity":       round(blb.sentiment.polarity, 4),
            "Subjectivity":   round(blb.sentiment.subjectivity, 4),
            "Dom_Emotion":    max(nrc, key=nrc.get) if any(nrc.values()) else "Neutral",
            **nrc,
        })
    return pd.DataFrame(rows)


# ── PER-MEETING FIGURE ────────────────────────────────────────────────────────
# Three panels: (A) valence bar  (B) NRC emotion profile  (C) emotional arc

def plot_meeting(df: pd.DataFrame, label: str, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle(f"Sentiment Summary — {label}", fontsize=14, fontweight="bold", y=1.01)

    # ── A: Valence distribution ──
    ax = axes[0]
    v_order  = ["Positive", "Neutral", "Negative"]
    v_counts = df["Valence"].value_counts().reindex(v_order, fill_value=0)
    v_colors = [VALENCE_COLORS[v] for v in v_order]
    bars = ax.bar(v_order, v_counts.values, color=v_colors,
                  edgecolor="white", linewidth=1.5, width=0.5)
    for bar, n in zip(bars, v_counts.values):
        pct = n / len(df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{n}  ({pct:.0f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    mean_c = df["Compound"].mean()
    ax.text(0.5, -0.22,
            f"Mean sentiment score: {mean_c:+.2f}   |   {len(df)} speaking turns",
            ha="center", transform=ax.transAxes, fontsize=9.5, color="#555")
    ax.set_title("A  |  Emotional Valence\n(how positive vs. negative is the language?)",
                 fontsize=10.5, loc="left", pad=6)
    ax.set_ylabel("Number of speaking turns")
    ax.set_ylim(0, v_counts.max() * 1.28)
    ax.tick_params(axis="x", labelsize=11)

    # ── B: NRC emotion profile ──
    ax = axes[1]
    means = df[EMOTION_COLS].mean().sort_values(ascending=True)
    colors_b = [EMOTION_COLORS[e] for e in means.index]
    bars = ax.barh(means.index, means.values, color=colors_b,
                   edgecolor="white", linewidth=1, height=0.6)
    for bar, val in zip(bars, means.values):
        ax.text(val + 0.004, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_title("B  |  Emotion Profile (Plutchik's 8 Emotions)\n(average proportion per speaking turn)",
                 fontsize=10.5, loc="left", pad=6)
    ax.set_xlabel("Mean proportion  (0 = absent → 1 = dominant)")
    ax.set_xlim(0, means.max() * 1.3)

    # ── C: Emotional arc ──
    ax = axes[2]
    x = df["Segment"].values
    y = df["Compound"].values
    w = max(5, len(x) // 10)        # window = ~10% of segments, minimum 5
    roll = pd.Series(y).rolling(w, center=True, min_periods=1).mean().values

    ax.fill_between(x, y, 0, where=(y >= 0), alpha=0.15, color="#27AE60")
    ax.fill_between(x, y, 0, where=(y <  0), alpha=0.20, color="#E74C3C")
    for xi, yi, val in zip(x, y, df["Valence"]):
        ax.scatter(xi, yi, color=VALENCE_COLORS[val], s=18, zorder=3, linewidths=0)
    ax.plot(x, roll, color="#2C3E50", linewidth=2, zorder=4,
            label=f"{w}-turn rolling avg")
    ax.axhline(0, color="#999", linewidth=0.8, linestyle="--")
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("Speaking turn (chronological)")
    ax.set_ylabel("Sentiment score\n(−1 = negative → +1 = positive)")
    ax.set_title("C  |  Emotional Arc\n(how sentiment shifts across the meeting)",
                 fontsize=10.5, loc="left", pad=6)
    leg = [mpatches.Patch(color="#27AE60", alpha=0.6, label="Positive"),
           mpatches.Patch(color="#E74C3C", alpha=0.6, label="Negative"),
           plt.Line2D([0],[0], color="#2C3E50", linewidth=2, label=f"{w}-turn avg")]
    ax.legend(handles=leg, fontsize=8.5, loc="lower right", framealpha=0.8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── CROSS-MEETING FIGURES ──────────────────────────────────────────────────────

def plot_cross_sentiment(summary: pd.DataFrame, out_path: Path):
    """
    Figure CM-1: Four key metrics compared across all meetings.
    Panels: mean compound score, % positive, % negative, mean subjectivity.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Sentiment Across All Meetings — Key Metrics",
                 fontsize=14, fontweight="bold", y=1.01)

    labels  = summary["Label"].tolist()
    x       = np.arange(len(labels))
    n_meet  = len(labels)

    # Shared x-axis formatting helper
    def fmt_ax(ax, title):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=10)
        ax.set_title(title, fontsize=11, loc="left", pad=6)
        ax.tick_params(axis="y", labelsize=10)

    # ── A: Mean VADER compound score ──
    ax = axes[0, 0]
    vals   = summary["Mean_Compound"].values
    colors = [VALENCE_COLORS["Positive"] if v >= 0.05
              else VALENCE_COLORS["Negative"] if v <= -0.05
              else VALENCE_COLORS["Neutral"]
              for v in vals]
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=1.5, width=0.55)
    ax.axhline(0, color="#999", linewidth=0.8, linestyle="--")
    for bar, v in zip(bars, vals):
        ypos = v + 0.01 if v >= 0 else v - 0.03
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{v:+.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    fmt_ax(ax, "A  |  Average Sentiment Score per Meeting\n(−1 = very negative → +1 = very positive)")
    ax.set_ylabel("Mean VADER compound score")
    ax.set_ylim(min(vals) - 0.15, max(vals) + 0.15)

    # ── B: % of turns that are positive ──
    ax = axes[0, 1]
    vals_pos = summary["Pct_Positive"].values
    bars = ax.bar(x, vals_pos, color="#27AE60", edgecolor="white", linewidth=1.5, width=0.55)
    for bar, v in zip(bars, vals_pos):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    fmt_ax(ax, "B  |  Percentage of Positive Speaking Turns")
    ax.set_ylabel("% of speaking turns")
    ax.set_ylim(0, 105)

    # ── C: % of turns that are negative ──
    ax = axes[1, 0]
    vals_neg = summary["Pct_Negative"].values
    bars = ax.bar(x, vals_neg, color="#E74C3C", edgecolor="white", linewidth=1.5, width=0.55)
    for bar, v in zip(bars, vals_neg):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    fmt_ax(ax, "C  |  Percentage of Negative Speaking Turns")
    ax.set_ylabel("% of speaking turns")
    ax.set_ylim(0, max(vals_neg) * 1.35)

    # ── D: Mean subjectivity ──
    ax = axes[1, 1]
    vals_sub = summary["Mean_Subjectivity"].values
    bars = ax.bar(x, vals_sub, color="#5B8DB8", edgecolor="white", linewidth=1.5, width=0.55)
    for bar, v in zip(bars, vals_sub):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    fmt_ax(ax, "D  |  Average Subjectivity of Language\n(0 = factual → 1 = personal/emotional)")
    ax.set_ylabel("Mean subjectivity score")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="#999", linewidth=0.8, linestyle="--")
    ax.text(n_meet - 0.5, 0.52, "midpoint", fontsize=8, color="#999")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cross_emotions(emotion_df: pd.DataFrame, out_path: Path):
    """
    Figure CM-2: NRC emotion profiles grouped by meeting (grouped bar chart)
    and dominant emotion composition (stacked bar).
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Emotion Profiles Across All Meetings",
                 fontsize=14, fontweight="bold", y=1.01)

    labels = emotion_df["Label"].tolist()
    n      = len(labels)
    x      = np.arange(n)

    # ── A: Grouped bar — mean proportion of each emotion per meeting ──
    ax = axes[0]
    n_emotions = len(EMOTION_COLS)
    width = 0.72 / n_emotions
    for i, emotion in enumerate(EMOTION_COLS):
        offset = (i - n_emotions / 2 + 0.5) * width
        vals   = emotion_df[f"AvgProportion_{emotion}"].values
        bars   = ax.bar(x + offset, vals, width=width * 0.88,
                        color=EMOTION_COLORS[emotion], edgecolor="white",
                        linewidth=0.6, label=emotion)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=10)
    ax.set_ylabel("Mean emotion proportion per speaking turn")
    ax.set_title("A  |  Emotion Intensity by Meeting\n(how much of each emotion appears in each meeting)",
                 fontsize=11, loc="left", pad=6)
    ax.legend(title="Emotion", fontsize=9, title_fontsize=9,
              loc="upper right", framealpha=0.85, ncol=2)
    avg_cols = [f"AvgProportion_{e}" for e in EMOTION_COLS]
    ax.set_ylim(0, emotion_df[avg_cols].values.max() * 1.35)

    # ── B: Stacked bar — dominant emotion composition per meeting ──
    ax = axes[1]
    bottom = np.zeros(n)
    for emotion in EMOTION_COLS:
        vals = emotion_df[f"Dom_Pct_{emotion}"].values
        ax.bar(x, vals, bottom=bottom, color=EMOTION_COLORS[emotion],
               edgecolor="white", linewidth=0.6, label=emotion)
        # Label slices that are large enough to be readable
        for xi, val, bot in zip(x, vals, bottom):
            if val > 4:
                ax.text(xi, bot + val / 2, f"{val:.0f}%",
                        ha="center", va="center", fontsize=8.5,
                        fontweight="bold", color="white")
        bottom += vals

    # Neutral remainder
    neutral_vals = emotion_df["Dom_Pct_Neutral"].values
    ax.bar(x, neutral_vals, bottom=bottom, color=EMOTION_COLORS["Neutral"],
           edgecolor="white", linewidth=0.6, label="Neutral")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=10)
    ax.set_ylabel("% of speaking turns")
    ax.set_ylim(0, 108)
    ax.set_title("B  |  Dominant Emotion Composition per Meeting\n(what emotion was most present in each speaking turn?)",
                 fontsize=11, loc="left", pad=6)
    handles = [mpatches.Patch(color=EMOTION_COLORS[e], label=e)
               for e in EMOTION_COLS + ["Neutral"]]
    ax.legend(handles=handles, title="Dominant Emotion",
              fontsize=9, title_fontsize=9, loc="upper right",
              framealpha=0.85, ncol=2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cross_arc(arc_df: pd.DataFrame, out_path: Path):
    """
    Figure CM-3: Emotional arc for all meetings overlaid on the same axes.
    X-axis = % through the meeting (so different-length meetings are comparable).
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    palette = sns.color_palette("tab10", n_colors=len(arc_df["Meeting"].unique()))

    for color, (label, group) in zip(palette, arc_df.groupby("Meeting", sort=False)):
        pct_x  = np.linspace(0, 100, len(group))
        smooth = pd.Series(group["Compound"].values).rolling(
            max(3, len(group) // 8), center=True, min_periods=1).mean().values
        ax.plot(pct_x, smooth, linewidth=2.2, color=color, label=label, alpha=0.9)

    ax.axhline(0, color="#999", linewidth=0.9, linestyle="--")
    ax.fill_between([0, 100], 0, 1.15,  alpha=0.04, color="#27AE60")
    ax.fill_between([0, 100], -1.15, 0, alpha=0.04, color="#E74C3C")
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Progress through meeting  (0% = start → 100% = end)", fontsize=11)
    ax.set_ylabel("Smoothed sentiment score\n(−1 = negative → +1 = positive)", fontsize=10)
    ax.set_title(
        "Emotional Arc — All Meetings Overlaid\n"
        "Each line is one meeting, smoothed for readability. "
        "Positive territory = above the dashed line.",
        fontsize=13, fontweight="bold", pad=10
    )
    ax.legend(title="Meeting", fontsize=10, title_fontsize=10,
              loc="lower right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    txt_files = sorted(MEETINGS_DIR.glob("*.txt"))
    if not txt_files:
        print("No .txt files found in", MEETINGS_DIR)
        return

    summary_rows = []
    emotion_rows = []
    arc_frames   = []

    for fpath in txt_files:
        label = meeting_label(fpath)
        print(f"Analyzing {label} ...")

        df = analyze_file(fpath)
        n  = len(df)
        if n == 0:
            print(f"  Skipped — no usable segments found.")
            continue

        # ── Per-meeting figure ──
        out_name = fpath.stem + "_summary.png"
        plot_meeting(df, label, GRAPHS_DIR / out_name)
        print(f"  Saved {out_name}  ({n} segments, {df['Word_Count'].sum():,} words)")

        # ── Accumulate summary stats ──
        vc = df["Valence"].value_counts()
        ic = df["Intensity"].value_counts()
        summary_rows.append({
            "Label":                    label,
            "Segments":                 n,
            "Total_Words":              df["Word_Count"].sum(),
            "Mean_Compound":            round(df["Compound"].mean(), 4),
            "Mean_Polarity":            round(df["Polarity"].mean(), 4),
            "Mean_Subjectivity":        round(df["Subjectivity"].mean(), 4),
            "Mean_VADER_Positive":      round(df["Positive_Pct"].mean(), 4),
            "Mean_VADER_Negative":      round(df["Negative_Pct"].mean(), 4),
            # Valence — counts and percentages
            "Count_Positive":           int(vc.get("Positive", 0)),
            "Count_Neutral":            int(vc.get("Neutral",  0)),
            "Count_Negative":           int(vc.get("Negative", 0)),
            "Pct_Positive":             round(vc.get("Positive", 0) / n * 100, 1),
            "Pct_Neutral":              round(vc.get("Neutral",  0) / n * 100, 1),
            "Pct_Negative":             round(vc.get("Negative", 0) / n * 100, 1),
            # Intensity — counts and percentages
            "Count_Strong":             int(ic.get("Strong",   0)),
            "Count_Moderate":           int(ic.get("Moderate", 0)),
            "Count_Mild":               int(ic.get("Mild",     0)),
            "Count_IntensityNeutral":   int(ic.get("Neutral",  0)),
            "Pct_Strong":               round(ic.get("Strong",   0) / n * 100, 1),
            "Pct_Moderate":             round(ic.get("Moderate", 0) / n * 100, 1),
            "Pct_Mild":                 round(ic.get("Mild",     0) / n * 100, 1),
            "Pct_IntensityNeutral":     round(ic.get("Neutral",  0) / n * 100, 1),
        })

        # ── Accumulate emotion stats ──
        dom_counts = df["Dom_Emotion"].value_counts()
        emotion_row = {"Label": label}
        emotion_row.update({f"AvgProportion_{e}": round(df[e].mean(), 4) for e in EMOTION_COLS})
        for e in EMOTION_COLS + ["Neutral"]:
            emotion_row[f"DomCount_{e}"]  = int(dom_counts.get(e, 0))
            emotion_row[f"Dom_Pct_{e}"]   = round(dom_counts.get(e, 0) / n * 100, 1)
        emotion_rows.append(emotion_row)

        # ── Accumulate full segment data (used for arc chart + segments CSV) ──
        arc_frames.append(df.assign(Meeting=label))

    if not summary_rows:
        print("No data to summarize.")
        return

    summary_df  = pd.DataFrame(summary_rows)
    emotion_df  = pd.DataFrame(emotion_rows)
    arc_df      = pd.concat(arc_frames, ignore_index=True)

    # ── Cross-meeting figures ──
    print("\nGenerating cross-meeting figures...")
    plot_cross_sentiment(summary_df, GRAPHS_DIR / "CM1_sentiment_comparison.png")
    print("  Saved CM1_sentiment_comparison.png")

    plot_cross_emotions(emotion_df, GRAPHS_DIR / "CM2_emotion_profiles.png")
    print("  Saved CM2_emotion_profiles.png")

    plot_cross_arc(arc_df, GRAPHS_DIR / "CM3_emotional_arcs_overlaid.png")
    print("  Saved CM3_emotional_arcs_overlaid.png")

    # ── Print summary table ──
    print("\n" + "=" * 68)
    print("SUMMARY TABLE")
    print("=" * 68)
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", None)
    print(summary_df.to_string(index=False))

    # Save summary CSV — merge sentiment + emotion counts/proportions
    combined_df = summary_df.merge(emotion_df.drop(columns=["Label"]), left_index=True, right_index=True)
    csv_summary = BASE_DIR / "all_meetings_summary.csv"
    combined_df.to_csv(csv_summary, index=False)
    print(f"\nSummary CSV saved to:  {csv_summary}")

    # Save full segment-level CSV (every speaking turn across all meetings)
    segments_df = pd.concat(arc_frames, ignore_index=True)
    # Reorder so Meeting is the first column
    col_order = ["Meeting"] + [c for c in segments_df.columns if c != "Meeting"]
    segments_df = segments_df[col_order]
    csv_segments = BASE_DIR / "all_meetings_segments.csv"
    segments_df.to_csv(csv_segments, index=False)
    print(f"Segments CSV saved to: {csv_segments}  ({len(segments_df)} rows)")
    print(f"\nAll graphs saved to:   {GRAPHS_DIR}")


if __name__ == "__main__":
    main()
