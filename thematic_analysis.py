"""
Focus Group — Thematic Analysis with Word Embeddings
=====================================================
Identifies recurring themes across all meetings using:
  - TF-IDF   : most distinctive words per meeting
  - LDA      : latent topic modeling across all meetings
  - Word2Vec : semantic word clusters via trained word embeddings + PCA

Outputs (saved to Graphs/):
  themes_fig1_distinctive_keywords.png  — top TF-IDF words per meeting
  themes_fig2_shared_keyword_heatmap.png — which keywords cross meeting boundaries
  themes_fig3_topic_model.png            — LDA topics: top words + meeting distribution
  themes_fig4_word_clusters.png          — Word2Vec 2D semantic cluster map
  themes_fig5_topic_wordclouds.png       — one word cloud per discovered topic

Required packages:
  pip install gensim scikit-learn wordcloud vaderSentiment textblob nrclex nltk pandas seaborn matplotlib
"""

import sys, io, re, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from gensim.models import Word2Vec

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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

# ── STYLE ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.family":      "sans-serif",
})

N_TOPICS    = 7     # number of LDA topics to extract
N_TOP_WORDS = 12    # top words shown per topic / per TF-IDF bar chart
MIN_WORDS   = 18    # minimum words per segment to include

# One distinct color per topic
TOPIC_PALETTE = [
    "#E74C3C", "#2980B9", "#27AE60", "#F39C12",
    "#8E44AD", "#1ABC9C", "#E67E22",
]

# ── STOP WORDS ────────────────────────────────────────────────────────────────
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
    "um", "okay", "yes", "no", "us", "get", "got", "also", "since",
    "though", "even", "really", "very", "quite", "bit", "lot",
    # transcript artifacts and discourse markers
    "towards", "table", "spot", "stuff", "point", "guess", "mean",
    "course", "example", "able", "start", "started", "talk", "talking",
    "happen", "happens", "happened", "end", "look", "looking", "next",
    "hey", "let", "anything", "apparently", "name", "else", "knowing",
    "interested", "different", "good", "hard", "okay", "clear", "big",
    "day", "week", "year", "time", "long", "high", "new", "last", "first",
    "whole", "part", "case", "type", "kinda", "gonna", "gotta", "wanna",
    "used", "using", "use", "using", "give", "given", "giving",
    "show", "shows", "showed", "bring", "brought", "keep", "kept",
    "move", "moved", "moving", "set", "sets", "run", "running",
}

lemmatizer = WordNetLemmatizer()


# ── HELPERS ───────────────────────────────────────────────────────────────────

def meeting_label(filepath: Path) -> str:
    stem  = filepath.stem
    parts = stem.split("_", 1)
    num   = parts[0]
    topic = parts[1].replace("_", " ") if len(parts) > 1 else ""
    topic = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", topic)
    return f"{num}: {topic}"


def parse_segments(filepath: Path) -> list[str]:
    text = filepath.read_text(encoding="utf-8", errors="replace")
    paras = re.split(r"\n\s*\n", text.strip())
    return [" ".join(p.split()) for p in paras[1:]
            if len(p.split()) >= MIN_WORDS]


def clean_tokens(text: str) -> list[str]:
    """Tokenize, lemmatize, remove stop words. Returns list of content words."""
    tokens = word_tokenize(text.lower())
    return [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t.isalpha() and len(t) > 2 and t not in STOP_WORDS
    ]


def tokens_to_str(tokens: list[str]) -> str:
    return " ".join(tokens)


# ── LOAD & PREPROCESS ─────────────────────────────────────────────────────────

print("Loading and preprocessing meetings...")

txt_files   = sorted(MEETINGS_DIR.glob("*.txt"))
labels      = [meeting_label(f) for f in txt_files]

# Per-meeting: list of cleaned segment strings
meeting_docs    = []   # one string per meeting (all segments joined)
meeting_segs    = []   # list-of-lists of token lists, per meeting
all_token_lists = []   # every segment as token list (for Word2Vec)

for fpath in txt_files:
    segs   = parse_segments(fpath)
    tokens = [clean_tokens(s) for s in segs]
    tokens = [t for t in tokens if len(t) >= 5]  # drop near-empty after cleaning

    meeting_segs.append(tokens)
    all_token_lists.extend(tokens)
    meeting_docs.append(tokens_to_str([w for tl in tokens for w in tl]))

print(f"  {len(txt_files)} meetings  |  {len(all_token_lists)} segments  |  "
      f"{sum(len(t) for t in all_token_lists):,} content words total")


# ── TF-IDF ────────────────────────────────────────────────────────────────────

print("Running TF-IDF...")

tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=2)
tfidf_matrix = tfidf.fit_transform(meeting_docs)   # shape: (n_meetings, vocab)
feature_names = np.array(tfidf.get_feature_names_out())


def top_tfidf_words(meeting_idx: int, n: int = N_TOP_WORDS):
    row    = tfidf_matrix[meeting_idx].toarray().flatten()
    top_ix = np.argsort(row)[::-1][:n]
    return [(feature_names[i], row[i]) for i in top_ix]


# ── LDA TOPIC MODELING ────────────────────────────────────────────────────────
# Use individual segments as documents (not meeting-level) so LDA has
# enough samples to discover meaningful topic structure.

print(f"Running LDA ({N_TOPICS} topics)...")

# Build one string per segment across all meetings, and track meeting membership
all_seg_strings  = []
seg_meeting_idx  = []
for m_idx, segs in enumerate(meeting_segs):
    for tlist in segs:
        if len(tlist) >= 5:
            all_seg_strings.append(tokens_to_str(tlist))
            seg_meeting_idx.append(m_idx)

count_vec = CountVectorizer(max_features=1500, ngram_range=(1, 1), min_df=3)
count_matrix = count_vec.fit_transform(all_seg_strings)
count_words  = np.array(count_vec.get_feature_names_out())

lda = LatentDirichletAllocation(
    n_components=N_TOPICS,
    max_iter=300,
    learning_method="batch",
    random_state=42,
)
lda.fit(count_matrix)

# Topic-word distributions (n_topics × vocab)
topic_word = lda.components_ / lda.components_.sum(axis=1, keepdims=True)

# Per-segment topic weights, then aggregate to meeting level
seg_topic_weights = lda.transform(count_matrix)   # (n_segs, n_topics)
meeting_topic_weights = np.zeros((len(labels), N_TOPICS))
meeting_seg_counts    = np.zeros(len(labels))
for seg_i, m_idx in enumerate(seg_meeting_idx):
    meeting_topic_weights[m_idx] += seg_topic_weights[seg_i]
    meeting_seg_counts[m_idx]    += 1
meeting_topic_weights /= meeting_seg_counts[:, None]   # average per meeting

# Top words for each topic
def top_topic_words(topic_idx: int, n: int = N_TOP_WORDS):
    top_ix = np.argsort(topic_word[topic_idx])[::-1][:n]
    return [(count_words[i], topic_word[topic_idx][i]) for i in top_ix]


# Generic single-syllable / short words to skip when labeling topics
GENERIC_LABEL_SKIP = {
    "day", "time", "way", "lot", "bit", "let", "hey", "yes", "use",
    "set", "run", "get", "got", "put", "end", "big", "new", "old",
    "good", "bad", "high", "low", "hard", "easy", "next", "last",
    "long", "short", "sure", "true", "else", "also", "back", "away",
}

def best_topic_label_word(topic_idx: int) -> str:
    """Return the highest-ranked word that is content-rich (len>4, not generic)."""
    for word, _ in top_topic_words(topic_idx, 20):
        if len(word) > 4 and word not in GENERIC_LABEL_SKIP:
            return word
    return top_topic_words(topic_idx, 1)[0][0]   # fallback

topic_labels = [f"Theme {i+1}: {best_topic_label_word(i).title()}"
                for i in range(N_TOPICS)]

print("  Discovered themes:")
for i, lbl in enumerate(topic_labels):
    top = ", ".join(w for w, _ in top_topic_words(i, 5))
    print(f"    {lbl}  —  {top}")


# ── WORD2VEC ──────────────────────────────────────────────────────────────────

print("Training Word2Vec embeddings...")

w2v = Word2Vec(
    sentences=all_token_lists,
    vector_size=100,
    window=6,
    min_count=4,
    workers=4,
    epochs=150,
    seed=42,
)

vocab    = list(w2v.wv.key_to_index.keys())
# Keep only reasonably frequent words for the cluster map
word_freq = Counter(w for tokens in all_token_lists for w in tokens)
plot_words = [w for w in vocab if word_freq[w] >= 8][:120]

vectors  = np.array([w2v.wv[w] for w in plot_words])

# PCA to 2D
pca     = PCA(n_components=2, random_state=42)
coords  = pca.fit_transform(vectors)

# K-Means clusters (same k as topics for consistency)
km      = KMeans(n_clusters=N_TOPICS, random_state=42, n_init=20)
cluster_ids = km.fit_predict(vectors)

print(f"  Vocabulary: {len(w2v.wv)} words  |  plotting top {len(plot_words)}")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 1 — TF-IDF Distinctive Keywords Per Meeting (2 × 3 grid)
# ─────────────────────────────────────────────────────────────────────────────

print("\nBuilding Figure 1: Distinctive keywords per meeting...")

fig1, axes = plt.subplots(2, 3, figsize=(18, 11))
fig1.suptitle(
    "Most Distinctive Keywords Per Meeting  (TF-IDF)\n"
    "Higher score = more unique to that meeting relative to the others",
    fontsize=14, fontweight="bold", y=1.01,
)

meeting_colors = sns.color_palette("tab10", n_colors=len(labels))

for idx, (ax, label, color) in enumerate(zip(axes.flat, labels, meeting_colors)):
    words_scores = top_tfidf_words(idx, N_TOP_WORDS)
    words  = [w for w, _ in reversed(words_scores)]
    scores = [s for _, s in reversed(words_scores)]

    # Color bars by score intensity
    norm_scores = np.array(scores) / max(scores)
    bar_colors  = [(*color[:3], 0.45 + 0.55 * v) for v in norm_scores]

    bars = ax.barh(words, scores, color=bar_colors, edgecolor="white",
                   linewidth=0.8, height=0.7)
    for bar, score in zip(bars, scores):
        ax.text(score + max(scores) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=8.5)

    ax.set_title(label, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("TF-IDF score", fontsize=9)
    ax.tick_params(axis="y", labelsize=9.5)
    ax.set_xlim(0, max(scores) * 1.22)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False)

plt.tight_layout()
out1 = GRAPHS_DIR / "themes_fig1_distinctive_keywords.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  Saved {out1.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 2 — Shared Keyword Heatmap (which words cross meeting boundaries?)
# ─────────────────────────────────────────────────────────────────────────────

print("Building Figure 2: Shared keyword heatmap...")

# Take the top 8 TF-IDF words from each meeting, collect unique set
top_words_per_meeting = [
    [w for w, _ in top_tfidf_words(i, 8)] for i in range(len(labels))
]
keyword_universe = list(dict.fromkeys(       # preserve first-seen order, deduplicate
    w for wlist in top_words_per_meeting for w in wlist
))

# Build matrix: meetings × keywords, value = TF-IDF score
tfidf_df = pd.DataFrame(
    index=labels,
    columns=keyword_universe,
    dtype=float,
)
for i, label in enumerate(labels):
    row = tfidf_matrix[i].toarray().flatten()
    for word in keyword_universe:
        wi = np.where(feature_names == word)[0]
        tfidf_df.loc[label, word] = float(row[wi[0]]) if len(wi) else 0.0

# Sort keywords so those appearing across most meetings come first
presence   = (tfidf_df > 0).sum(axis=0)
tfidf_df   = tfidf_df[presence.sort_values(ascending=False).index]

fig2, ax = plt.subplots(figsize=(20, 6))
sns.heatmap(
    tfidf_df,
    ax=ax,
    cmap="YlOrBr",
    linewidths=0.4,
    linecolor="#eeeeee",
    cbar_kws={"label": "TF-IDF score  (0 = absent, higher = more distinctive)",
              "shrink": 0.65},
    annot=tfidf_df.round(3),
    annot_kws={"size": 8},
    fmt="",
)
ax.set_title(
    "Keyword Presence Across Meetings\n"
    "Words toward the left appear in more meetings (shared themes); "
    "words toward the right are meeting-specific",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("Keyword", fontsize=11)
ax.set_ylabel("Meeting", fontsize=11)
ax.tick_params(axis="x", rotation=40, labelsize=9)
ax.tick_params(axis="y", rotation=0, labelsize=10)

plt.tight_layout()
out2 = GRAPHS_DIR / "themes_fig2_shared_keyword_heatmap.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved {out2.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 3 — LDA Topic Model  (top words + meeting distribution)
# ─────────────────────────────────────────────────────────────────────────────

print("Building Figure 3: LDA topic model...")

fig3 = plt.figure(figsize=(20, 13))
fig3.suptitle(
    f"Latent Topic Model  ({N_TOPICS} Themes Discovered via LDA)\n"
    "Left: defining words for each theme  |  Right: how much each theme appears in each meeting",
    fontsize=13, fontweight="bold", y=1.01,
)

gs = GridSpec(N_TOPICS, 5, figure=fig3, wspace=0.6, hspace=0.55)

# Left side: one horizontal bar chart per topic (occupying first 3 columns)
for t_idx in range(N_TOPICS):
    ax = fig3.add_subplot(gs[t_idx, :3])
    words_scores = top_topic_words(t_idx, N_TOP_WORDS)
    words  = [w for w, _ in reversed(words_scores)]
    scores = [s for _, s in reversed(words_scores)]

    norm   = np.array(scores) / max(scores)
    base   = matplotlib.colors.to_rgb(TOPIC_PALETTE[t_idx])
    colors = [(*base, 0.4 + 0.6 * v) for v in norm]

    ax.barh(words, scores, color=colors, edgecolor="white",
            linewidth=0.6, height=0.7)
    ax.set_title(topic_labels[t_idx], fontsize=10, fontweight="bold",
                 color=TOPIC_PALETTE[t_idx], pad=3)
    ax.set_xlim(0, max(scores) * 1.15)
    ax.tick_params(axis="y", labelsize=8.5, left=False)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.spines["left"].set_visible(False)
    if t_idx < N_TOPICS - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Word probability within theme", fontsize=8.5)

# Right side: heatmap of meeting × topic weights (last 2 columns)
ax_heat = fig3.add_subplot(gs[:, 3:])
heat_data = pd.DataFrame(
    meeting_topic_weights,
    index=labels,
    columns=[f"Theme {i+1}" for i in range(N_TOPICS)],
)
sns.heatmap(
    heat_data.T,
    ax=ax_heat,
    cmap="Blues",
    annot=True, fmt=".2f",
    annot_kws={"size": 9},
    linewidths=0.5,
    linecolor="#eeeeee",
    cbar_kws={"label": "Theme weight  (higher = more present in this meeting)",
              "shrink": 0.6},
)
ax_heat.set_title("Theme Presence\nPer Meeting", fontsize=11,
                  fontweight="bold", pad=8)
ax_heat.set_xlabel("Meeting", fontsize=10)
ax_heat.set_ylabel("Theme", fontsize=10)
ax_heat.tick_params(axis="x", rotation=30, labelsize=9)
ax_heat.tick_params(axis="y", rotation=0, labelsize=9)

out3 = GRAPHS_DIR / "themes_fig3_topic_model.png"
fig3.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"  Saved {out3.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 4 — Word2Vec Semantic Cluster Map
# ─────────────────────────────────────────────────────────────────────────────

print("Building Figure 4: Word embedding cluster map...")

fig4, ax = plt.subplots(figsize=(16, 12))

# Draw each cluster as a scatter + convex hull shading
for c_id in range(N_TOPICS):
    mask = cluster_ids == c_id
    if mask.sum() < 2:
        continue
    ax.scatter(
        coords[mask, 0], coords[mask, 1],
        color=TOPIC_PALETTE[c_id], s=60, alpha=0.85,
        edgecolors="white", linewidths=0.6, zorder=3,
        label=f"Cluster {c_id + 1}",
    )
    # Convex hull shading
    pts = coords[mask]
    if len(pts) >= 3:
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            poly = plt.Polygon(hull_pts, closed=True,
                               facecolor=TOPIC_PALETTE[c_id],
                               alpha=0.07, edgecolor=TOPIC_PALETTE[c_id],
                               linewidth=1, linestyle="--")
            ax.add_patch(poly)
        except Exception:
            pass

# Word labels — stagger slightly to reduce overlap
rng = np.random.default_rng(0)
for word, (x, y), c_id in zip(plot_words, coords, cluster_ids):
    jitter_x = rng.uniform(-0.01, 0.01) * ((coords[:, 0].max() - coords[:, 0].min()))
    jitter_y = rng.uniform(-0.01, 0.01) * ((coords[:, 1].max() - coords[:, 1].min()))
    ax.text(
        x + jitter_x, y + jitter_y, word,
        fontsize=8, ha="center", va="center",
        color=TOPIC_PALETTE[c_id],
        fontweight="bold" if word_freq[word] >= 20 else "normal",
        alpha=0.90,
    )

ax.set_title(
    "Semantic Word Cluster Map  (Word2Vec Embeddings → PCA 2D)\n"
    "Words placed close together are used in similar contexts across all meetings.\n"
    "Color = semantic cluster  |  Bold = high-frequency words",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel(f"PCA Component 1  ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=10)
ax.set_ylabel(f"PCA Component 2  ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=10)
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax.set_facecolor("#FAFAFA")

legend_handles = [
    mpatches.Patch(color=TOPIC_PALETTE[c], label=f"Cluster {c+1}")
    for c in range(N_TOPICS)
]
ax.legend(handles=legend_handles, title="Semantic Cluster",
          fontsize=9.5, title_fontsize=10, loc="lower right",
          framealpha=0.9, edgecolor="#ccc")

plt.tight_layout()
out4 = GRAPHS_DIR / "themes_fig4_word_clusters.png"
fig4.savefig(out4, dpi=150, bbox_inches="tight")
plt.close(fig4)
print(f"  Saved {out4.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 5 — Word Clouds Per Topic
# ─────────────────────────────────────────────────────────────────────────────

print("Building Figure 5: Topic word clouds...")

rows_wc = 2 if N_TOPICS <= 6 else 2
cols_wc = (N_TOPICS + 1) // 2

fig5, axes = plt.subplots(rows_wc, cols_wc,
                           figsize=(cols_wc * 5.5, rows_wc * 4))
fig5.suptitle(
    "Word Clouds Per Discovered Theme\n"
    "Word size = relative importance within that theme",
    fontsize=14, fontweight="bold", y=1.01,
)

for t_idx, ax in enumerate(axes.flat):
    if t_idx >= N_TOPICS:
        ax.axis("off")
        continue

    # Build frequency dict from topic-word probabilities
    word_probs = {count_words[i]: float(topic_word[t_idx, i])
                  for i in range(len(count_words))}

    base_color = TOPIC_PALETTE[t_idx]

    def color_func(*args, **kwargs):
        r, g, b = matplotlib.colors.to_rgb(base_color)
        # Vary lightness slightly so cloud isn't monochrome
        factor = 0.7 + np.random.rand() * 0.3
        return (int(r * 255 * factor), int(g * 255 * factor), int(b * 255 * factor))

    wc = WordCloud(
        width=600, height=380,
        background_color="white",
        max_words=60,
        color_func=color_func,
        prefer_horizontal=0.85,
        random_state=t_idx,
        collocations=False,
    ).generate_from_frequencies(word_probs)

    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(topic_labels[t_idx], fontsize=11,
                 fontweight="bold", color=base_color, pad=6)

plt.tight_layout()
out5 = GRAPHS_DIR / "themes_fig5_topic_wordclouds.png"
fig5.savefig(out5, dpi=150, bbox_inches="tight")
plt.close(fig5)
print(f"  Saved {out5.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  DONE
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nAll theme figures saved to: {GRAPHS_DIR}")
print("\nFigure guide:")
print("  themes_fig1_distinctive_keywords.png  — top TF-IDF words unique to each meeting")
print("  themes_fig2_shared_keyword_heatmap.png — which keywords appear across multiple meetings")
print("  themes_fig3_topic_model.png            — 7 latent themes: defining words + meeting weights")
print("  themes_fig4_word_clusters.png          — semantic neighborhoods in 2D word space")
print("  themes_fig5_topic_wordclouds.png       — word cloud for each discovered theme")
