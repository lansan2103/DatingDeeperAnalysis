"""
Focus Group Sentiment — Visualization Suite
=============================================
Reads sentiment_analysis_output.csv and produces five publication-ready figures.
Run after sentiment_analysis.py.

Required packages:  pip install seaborn matplotlib pandas
"""

import sys
import io
import warnings
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─── PATHS ────────────────────────────────────────────────────────────────────
CSV_PATH    = r"c:\Users\never\OneDrive\Desktop\D&D Sentiment Analysis\sentiment_analysis_output.csv"
OUTPUT_DIR  = r"c:\Users\never\OneDrive\Desktop\D&D Sentiment Analysis"

# ─── GLOBAL STYLE ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor":   "#FAFAFA",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.family":      "sans-serif",
})

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

VALENCE_COLORS = {
    "Positive": "#27AE60",
    "Neutral":  "#BDC3C7",
    "Negative": "#E74C3C",
}

INTENSITY_ORDER  = ["Strong", "Moderate", "Mild", "Neutral"]
EMOTION_COLS     = ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation"]

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# ─── FIGURE 1: OVERVIEW DASHBOARD ─────────────────────────────────────────────
# 2x2 grid: valence distribution, intensity distribution,
#           NRC emotion profile, polarity vs. subjectivity scatter

fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle(
    "Focus Group Sentiment Overview\nD&D Recording — Summary Dashboard",
    fontsize=14, fontweight="bold", y=1.01
)

# ── Panel A: Valence distribution ──
ax = axes[0, 0]
valence_order = ["Positive", "Neutral", "Negative"]
valence_counts = df["Overall_Emotional_Valence"].value_counts().reindex(valence_order, fill_value=0)
colors_v = [VALENCE_COLORS[v] for v in valence_order]
bars = ax.bar(valence_order, valence_counts.values, color=colors_v, edgecolor="white", linewidth=1.5, width=0.55)
for bar, count in zip(bars, valence_counts.values):
    pct = count / len(df) * 100
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f"{count}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("A  |  Overall Emotional Valence\nHow positive or negative is the language?",
             fontsize=11, loc="left", pad=8)
ax.set_ylabel("Number of Speaking Turns")
ax.set_ylim(0, max(valence_counts.values) * 1.2)
ax.tick_params(axis="x", labelsize=11)

# ── Panel B: Emotional intensity distribution ──
ax = axes[0, 1]
intensity_counts = df["Emotional_Intensity"].value_counts().reindex(INTENSITY_ORDER, fill_value=0)
intensity_palette = ["#E74C3C", "#E67E22", "#F1C40F", "#BDC3C7"]
bars = ax.bar(INTENSITY_ORDER, intensity_counts.values,
              color=intensity_palette, edgecolor="white", linewidth=1.5, width=0.55)
for bar, count in zip(bars, intensity_counts.values):
    pct = count / len(df) * 100
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{count}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("B  |  Emotional Intensity Distribution\nHow strongly is emotion expressed?",
             fontsize=11, loc="left", pad=8)
ax.set_ylabel("Number of Speaking Turns")
ax.set_ylim(0, max(intensity_counts.values) * 1.2)

# ── Panel C: NRC emotion profile (mean proportions) ──
ax = axes[1, 0]
emotion_means = df[EMOTION_COLS].mean().sort_values(ascending=True)
bar_colors = [EMOTION_COLORS[e] for e in emotion_means.index]
bars = ax.barh(emotion_means.index, emotion_means.values,
               color=bar_colors, edgecolor="white", linewidth=1.2, height=0.6)
for bar, val in zip(bars, emotion_means.values):
    ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9)
ax.set_title("C  |  NRC Emotion Profile (Mean Proportion)\nPlutchik's (1980) Wheel — averaged across all turns",
             fontsize=11, loc="left", pad=8)
ax.set_xlabel("Mean Proportion (0 = absent, 1 = all words)")
ax.set_xlim(0, emotion_means.max() * 1.25)

# ── Panel D: Polarity vs. Subjectivity scatter ──
ax = axes[1, 1]
dom_emotion_list = df["Dominant_Emotion"].tolist()
point_colors = [EMOTION_COLORS.get(e, "#BDC3C7") for e in dom_emotion_list]
scatter = ax.scatter(df["Subjectivity_Score"], df["Polarity_Score"],
                     c=point_colors, s=55, alpha=0.75, edgecolors="white", linewidths=0.5)
ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
ax.axvline(0.5, color="#999999", linewidth=0.8, linestyle="--")
ax.set_xlabel("Subjectivity  (0 = Factual  →  1 = Personal/Emotional)")
ax.set_ylabel("Polarity  (−1 = Negative  →  +1 = Positive)")
ax.set_title("D  |  Polarity vs. Subjectivity\nColored by dominant emotion",
             fontsize=11, loc="left", pad=8)
# Quadrant labels
ax.text(0.03, 0.97, "Objective\nNegative",  transform=ax.transAxes, fontsize=8, color="#888", va="top")
ax.text(0.70, 0.97, "Personal\nNegative",   transform=ax.transAxes, fontsize=8, color="#888", va="top")
ax.text(0.03, 0.06, "Objective\nPositive",  transform=ax.transAxes, fontsize=8, color="#888", va="bottom")
ax.text(0.70, 0.06, "Personal\nPositive",   transform=ax.transAxes, fontsize=8, color="#888", va="bottom")
# Compact legend for dominant emotions actually present
present_emotions = df["Dominant_Emotion"].unique()
legend_handles = [mpatches.Patch(color=EMOTION_COLORS.get(e, "#BDC3C7"), label=e)
                  for e in EMOTION_COLS + ["Neutral"] if e in present_emotions]
ax.legend(handles=legend_handles, title="Dominant Emotion",
          fontsize=8, title_fontsize=8, loc="lower right",
          framealpha=0.7, ncol=2)

plt.tight_layout()
out1 = f"{OUTPUT_DIR}\\fig1_overview_dashboard.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved: {out1}")


# ─── FIGURE 2: EMOTIONAL ARC ──────────────────────────────────────────────────
# VADER compound score across all segments with rolling average and color-coded fill

fig2, ax = plt.subplots(figsize=(16, 5))
x = df["Segment_Number"].values
y = df["VADER_Compound_Score"].values
window = 7
rolling = pd.Series(y).rolling(window, center=True, min_periods=1).mean().values

# Fill positive / negative regions
ax.fill_between(x, y, 0, where=(y >= 0), alpha=0.15, color="#27AE60", label="_nolegend_")
ax.fill_between(x, y, 0, where=(y < 0),  alpha=0.20, color="#E74C3C", label="_nolegend_")

# Raw segment dots colored by valence
for xi, yi, val in zip(x, y, df["Overall_Emotional_Valence"]):
    ax.scatter(xi, yi, color=VALENCE_COLORS[val], s=22, zorder=3, linewidths=0)

# Rolling average line
ax.plot(x, rolling, color="#2C3E50", linewidth=2.2, zorder=4,
        label=f"{window}-segment rolling average")

ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel("Speaking Turn (chronological order)", fontsize=11)
ax.set_ylabel("VADER Compound Score\n(−1 = Very Negative  →  +1 = Very Positive)", fontsize=10)
ax.set_title(
    "Emotional Arc Across the Focus Group\n"
    "Each dot = one speaking turn; line = smoothed trend",
    fontsize=13, fontweight="bold", pad=10
)

# Annotation: where are major dips?
min_idx = df["VADER_Compound_Score"].idxmin()
ax.annotate(
    f"Most negative turn\n(Seg {int(df.loc[min_idx,'Segment_Number'])})",
    xy=(df.loc[min_idx, "Segment_Number"], df.loc[min_idx, "VADER_Compound_Score"]),
    xytext=(df.loc[min_idx, "Segment_Number"] + 4, -0.75),
    arrowprops=dict(arrowstyle="->", color="#555"),
    fontsize=9, color="#555"
)

legend_patches = [
    mpatches.Patch(color="#27AE60", alpha=0.6, label="Positive turn"),
    mpatches.Patch(color="#E74C3C", alpha=0.6, label="Negative turn"),
    mpatches.Patch(color="#BDC3C7", alpha=0.8, label="Neutral turn"),
    plt.Line2D([0], [0], color="#2C3E50", linewidth=2.2, label=f"{window}-turn rolling average"),
]
ax.legend(handles=legend_patches, loc="lower right", fontsize=9, framealpha=0.8)

plt.tight_layout()
out2 = f"{OUTPUT_DIR}\\fig2_emotional_arc.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")


# ─── FIGURE 3: NRC EMOTION HEATMAP ────────────────────────────────────────────
# Rows = emotions, columns = speaking segments; colour = emotion proportion

fig3, ax = plt.subplots(figsize=(18, 5))

heatmap_data = df[EMOTION_COLS].T  # shape: (8 emotions × 102 segments)

sns.heatmap(
    heatmap_data,
    ax=ax,
    cmap="YlOrRd",
    linewidths=0,
    cbar_kws={"label": "Emotion Proportion", "shrink": 0.6},
    xticklabels=10,   # show every 10th segment label to avoid crowding
    yticklabels=True,
    vmin=0, vmax=0.6,
)
ax.set_xlabel("Speaking Turn Number", fontsize=11)
ax.set_ylabel("Emotion", fontsize=11)
ax.set_title(
    "NRC Emotion Heatmap Across the Focus Group\n"
    "Darker = that emotion more strongly present in that speaking turn",
    fontsize=13, fontweight="bold", pad=10
)
ax.tick_params(axis="y", labelsize=10)

plt.tight_layout()
out3 = f"{OUTPUT_DIR}\\fig3_emotion_heatmap.png"
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved: {out3}")


# ─── FIGURE 4: DOMINANT EMOTION BREAKDOWN ─────────────────────────────────────
# Stacked: count plot + proportion of compound score within each emotion category

fig4, axes = plt.subplots(1, 2, figsize=(14, 6))
fig4.suptitle(
    "Dominant Emotion Analysis\nFrequency and Emotional Tone per Emotion Category",
    fontsize=13, fontweight="bold"
)

# ── Panel A: Frequency count ──
ax = axes[0]
dom_order = (df["Dominant_Emotion"].value_counts().index.tolist())
dom_counts = df["Dominant_Emotion"].value_counts()
bar_cols = [EMOTION_COLORS.get(e, "#BDC3C7") for e in dom_order]
bars = ax.bar(dom_order, dom_counts.values, color=bar_cols,
              edgecolor="white", linewidth=1.5, width=0.6)
for bar, count in zip(bars, dom_counts.values):
    pct = count / len(df) * 100
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{count}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_title("A  |  How often was each emotion dominant?", fontsize=11, loc="left", pad=6)
ax.set_ylabel("Number of Speaking Turns")
ax.set_xlabel("Dominant Emotion")
ax.tick_params(axis="x", labelsize=10)
ax.set_ylim(0, dom_counts.max() * 1.25)

# ── Panel B: Box plot of VADER score per dominant emotion ──
ax = axes[1]
palette_b = {e: EMOTION_COLORS.get(e, "#BDC3C7") for e in dom_order}
sns.boxplot(
    data=df, x="Dominant_Emotion", y="VADER_Compound_Score",
    order=dom_order, palette=palette_b,
    linewidth=1.2, fliersize=4, ax=ax,
    medianprops=dict(color="black", linewidth=2),
)
ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
ax.set_title("B  |  Emotional Valence within Each Dominant Emotion\n"
             "(Box = middle 50% of scores; line = median)", fontsize=11, loc="left", pad=6)
ax.set_xlabel("Dominant Emotion")
ax.set_ylabel("VADER Compound Score\n(−1 = Negative  →  +1 = Positive)")
ax.tick_params(axis="x", labelsize=10)

plt.tight_layout()
out4 = f"{OUTPUT_DIR}\\fig4_dominant_emotion_breakdown.png"
fig4.savefig(out4, dpi=150, bbox_inches="tight")
print(f"Saved: {out4}")


# ─── FIGURE 5: EMOTION CORRELATION MATRIX ────────────────────────────────────
# How often do emotions co-occur in the same speaking turn?

fig5, ax = plt.subplots(figsize=(8, 7))

corr = df[EMOTION_COLS].corr()

mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # hide upper triangle

sns.heatmap(
    corr,
    ax=ax,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5,
    linecolor="#EEEEEE",
    cbar_kws={"label": "Pearson r", "shrink": 0.75},
    annot_kws={"size": 9},
)
ax.set_title(
    "Emotion Co-occurrence Correlation Matrix\n"
    "How often do pairs of emotions appear together in the same turn?\n"
    "(+1 = always together, -1 = never together, 0 = unrelated)",
    fontsize=12, fontweight="bold", pad=12
)
ax.tick_params(axis="x", rotation=45, labelsize=10)
ax.tick_params(axis="y", rotation=0,  labelsize=10)

plt.tight_layout()
out5 = f"{OUTPUT_DIR}\\fig5_emotion_correlation.png"
fig5.savefig(out5, dpi=150, bbox_inches="tight")
print(f"Saved: {out5}")


# ─── DONE ─────────────────────────────────────────────────────────────────────
print("\nAll figures saved to:", OUTPUT_DIR)
print("\nFigure guide:")
print("  fig1_overview_dashboard.png    — 4-panel summary (valence, intensity, emotion profile, scatter)")
print("  fig2_emotional_arc.png         — Sentiment trajectory across the full conversation")
print("  fig3_emotion_heatmap.png       — All 8 emotions across all 102 speaking turns")
print("  fig4_dominant_emotion_breakdown.png — Frequency + valence range per dominant emotion")
print("  fig5_emotion_correlation.png   — Which emotions tend to co-occur?")
