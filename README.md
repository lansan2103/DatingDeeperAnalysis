# D&D Focus Group — Computational Sentiment & Thematic Analysis

## Overview

This project applies computational natural language processing (NLP) to a series of focus group transcripts. The goal is to systematically quantify emotional content, sentiment patterns, and recurring themes across conversations — producing outputs that complement traditional qualitative coding.

All scripts are written in Python. No coding background is required to interpret the outputs; each graph and CSV column is labeled for a psychology research audience.

---

## Project Files

| File | Purpose |
|------|---------|
| `sentiment_analysis.py` | Sentiment analysis on a single transcript |
| `visualizations.py` | Visualization suite for the single transcript |
| `analyze_all_meetings.py` | Sentiment analysis across all six meeting transcripts |
| `thematic_analysis.py` | Thematic/topic analysis using word embeddings |
| `all_meetings_summary.csv` | Aggregated sentiment + emotion counts per meeting |
| `all_meetings_segments.csv` | Full segment-level data for all meetings (627 rows) |
| `Graphs/` | All generated figures |

---

## How to Run

Install dependencies once:
```
pip install vaderSentiment textblob nrclex nltk pandas seaborn matplotlib gensim scikit-learn wordcloud
```

Then run scripts in order:
```
py analyze_all_meetings.py    # produces CSVs + per-meeting and cross-meeting graphs
py thematic_analysis.py       # produces thematic/word embedding graphs
```

---

## Data Preprocessing

Before any model runs, each transcript goes through a shared preprocessing pipeline. Understanding these steps is important for interpreting what the models "see."

### Segmentation
Each transcript is split into speaking turns — paragraphs separated by blank lines. Very short fragments (fewer than 15–18 words) are discarded, as they typically represent backchannels ("yeah," "mm-hmm"), crosstalk, or incomplete sentences that carry no analyzable content.

### Stop Word Removal
Stop words are common words that carry little semantic meaning on their own. Two layers are removed:

1. **Standard English stop words** — function words such as *the, is, at, which, on* (from NLTK's built-in list).
2. **Transcript-specific filler words** — spoken language patterns that appear frequently but are not substantively meaningful in this context, such as *like, know, yeah, just, kind of, sort of, actually, basically, totally, honestly*. These are especially common in informal group conversation and would otherwise dominate any frequency-based analysis.

This ensures that the models focus on content words — nouns, meaningful verbs, adjectives — that actually carry psychological and thematic information.

### Lemmatization (Thematic Analysis only)
In `thematic_analysis.py`, words are further reduced to their base form (lemma) before topic modeling and embedding. For example, *relationships*, *relating*, and *related* all become *relation*. This reduces vocabulary noise and helps models recognise that the same concept is being discussed even when expressed in slightly different grammatical forms.

---

## Sentiment Analysis Models

### 1. VADER (Valence Aware Dictionary and sEntiment Reasoner)

**What it does:** VADER assigns each piece of text a sentiment score on a scale from −1.0 (maximally negative) to +1.0 (maximally positive), called the **compound score**. It also reports what proportion of the text's words are positive, negative, or neutral.

**How it works:** VADER uses a hand-curated lexicon of over 7,500 words and phrases, each pre-rated for emotional valence by human annotators. It has special rules for spoken and informal language: it recognises that capitalisation (*GREAT* vs *great*), punctuation (*great!!!*), and intensifiers (*very good*, *kind of bad*) all modify the emotional weight of a word. Crucially, it also handles negation — *not happy* is scored as negative, not positive.

**Why it was chosen:** VADER was specifically designed and validated for conversational, informal text — the same register as focus group speech. It outperforms general-purpose models on social media and interview data. It requires no training data and runs instantly, making it ideal for exploratory clinical research.

**Outputs:**
- `VADER_Compound_Score` — overall emotional valence (−1 to +1)
- `VADER_Positive_Content` — proportion of positively-valenced words (0–1)
- `VADER_Negative_Content` — proportion of negatively-valenced words (0–1)
- `Overall_Emotional_Valence` — simplified label: Positive / Neutral / Negative
- `Emotional_Intensity` — strength label: Strong / Moderate / Mild / Neutral

**Interpretation note:** VADER scores the surface language, not the speaker's subjective experience. A participant describing their own depression in matter-of-fact, even hopeful terms may score as neutral or positive. Compound scores should be interpreted alongside the actual excerpt and NRC emotion data.

---

### 2. TextBlob Polarity and Subjectivity

**What it does:** TextBlob provides two independent scores for each segment:
- **Polarity** (−1.0 to +1.0): similar to VADER's compound score; how positive or negative the language is.
- **Subjectivity** (0.0 to 1.0): how personal and opinion-based the language is, versus factual and objective.

**How it works:** TextBlob uses a pattern-based lexicon trained on movie reviews and other opinionated text. Each word in its lexicon carries both a polarity rating and a subjectivity rating. The scores for a sentence are averaged across its constituent words, with some adjustments for modifiers.

**Why it was chosen:** VADER does not measure subjectivity. TextBlob's subjectivity score adds a dimension that VADER lacks — it captures not just whether language is emotionally positive or negative, but how emotionally *engaged* or *personal* the speaker's language is at all. In focus group research, high subjectivity often corresponds to personal disclosure, which is theoretically significant (e.g., self-disclosure in group settings, vulnerability).

**Outputs:**
- `Polarity_Score` — language tone (−1 = negative, +1 = positive)
- `Subjectivity_Score` — language register (0 = factual/clinical, 1 = personal/emotional)

---

### 3. NRC Emotion Lexicon (Plutchik's Wheel of Emotions)

**What it does:** The NRC lexicon maps individual words to one or more of eight basic emotions: **Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger,** and **Anticipation**. For each speaking turn, it reports what proportion of emotion-bearing words belong to each category.

**How it works:** The NRC Word-Emotion Association Lexicon (Mohammad & Turney, 2013) was constructed through crowd-sourced annotation of over 14,000 English words, where annotators indicated which of Plutchik's eight emotions (and positive/negative valence) each word evoked. For a given text, the model counts how many words match each emotion category, then normalises to proportions. The dominant emotion — whichever category has the highest proportion — is reported as a label.

**Why it was chosen:** Plutchik's Wheel of Emotions (1980) is a well-established theoretical framework in psychology. By anchoring the computational model to this framework, the outputs are directly interpretable within existing psychological literature on emotion. Unlike VADER (which only measures positive/negative), NRC distinguishes *qualitatively different* emotional states — for example, a segment can score high on both Fear and Trust simultaneously (ambivalence), or high on Sadness without scoring negative on VADER (e.g., someone describing grief in a measured, reflective tone).

**Outputs (per segment and per meeting):**
- `Joy`, `Trust`, `Fear`, `Surprise`, `Sadness`, `Disgust`, `Anger`, `Anticipation` — proportional scores (0–1; sum ≈ 1.0)
- `Dominant_Emotion` — the highest-scoring category for that segment
- `DomCount_[Emotion]` — number of turns where that emotion was dominant
- `Dom_Pct_[Emotion]` — percentage of turns where that emotion was dominant
- `AvgProportion_[Emotion]` — mean proportion across all turns in the meeting

---

## Thematic Analysis Models

### 4. TF-IDF (Term Frequency–Inverse Document Frequency)

**What it does:** TF-IDF identifies the words that are most *distinctive* to each meeting — not simply the most common words overall, but the words that are unusually frequent in one meeting relative to all the others.

**How it works:** For any word in a given document (meeting):
- **Term Frequency (TF):** how often the word appears in this meeting.
- **Inverse Document Frequency (IDF):** a penalty applied if the word appears in many meetings. Words that appear in every meeting receive a very low IDF score because they are not distinctive to any one of them.

The final TF-IDF score = TF × IDF. A high score means the word is both frequent in this meeting *and* rare in the others — a strong signal of meeting-specific vocabulary.

**Why it was chosen:** Raw word frequency would just return generic conversation words (after stop word removal, terms like *relationship* or *friend* would still dominate every meeting). TF-IDF reveals what is *unique* about each meeting's discourse — for example, if a specific meeting uses the word *conflict* much more than the others, TF-IDF will flag it. This is directly analogous to what a qualitative researcher does when noting that certain topics or constructs are more prominent in particular sessions.

**Outputs:**
- `themes_fig1_distinctive_keywords.png` — top 12 distinctive words per meeting (bar charts)
- `themes_fig2_shared_keyword_heatmap.png` — TF-IDF scores across all meetings, showing which words are shared vs. session-specific

---

### 5. LDA (Latent Dirichlet Allocation) Topic Modeling

**What it does:** LDA is an unsupervised machine learning algorithm that reads all segments across all meetings and discovers a specified number of hidden themes — groups of words that tend to appear together. It does this without any human guidance about what those themes should be.

**How it works:** LDA treats each segment as a mixture of topics, and each topic as a probability distribution over words. For example, a topic about mental health might assign high probability to words like *depression*, *anxiety*, *therapy*, and *support*. The algorithm iterates until it finds topic assignments that best explain which words co-occur across segments. The output is:
1. For each topic: a ranked list of words most associated with it.
2. For each segment: a probability distribution across all topics (how much of this segment belongs to each theme).
3. For each meeting: averaged topic weights, showing which themes dominated that session.

**Why it was chosen:** LDA is one of the most widely used methods for theme discovery in qualitative and mixed-methods health research. It makes no assumptions about what the themes will be — it lets the data determine the structure. This makes it well-suited as a complement to theory-driven coding, revealing patterns that might not have been anticipated. Running LDA on individual *segments* (not on meetings as a whole) gives it sufficient data points to distinguish meaningful thematic variation.

**Model parameters:**
- **7 topics** — chosen to roughly match the number of distinct discussion themes across six meetings, with some overlap expected
- **300 iterations** — sufficient for convergence on a corpus of this size
- `min_df=3` — a word must appear in at least 3 segments to enter the model, preventing rare or idiosyncratic words from distorting topics

**Outputs:**
- `themes_fig3_topic_model.png` — left panel: top 12 words per topic; right panel: heatmap of theme weights per meeting
- `themes_fig5_topic_wordclouds.png` — word cloud per topic, where word size reflects importance within that theme

**Interpretation note:** LDA topics are unlabeled — the algorithm does not know what a theme "means." The researcher is expected to read the top words for each topic and assign a meaningful label based on psychological or theoretical judgment. The word clouds are designed to make this interpretation as intuitive as possible.

---

### 6. Word2Vec Word Embeddings

**What it does:** Word2Vec trains a neural network to learn dense numerical representations (vectors) for every word in the corpus, such that words used in similar conversational contexts end up close together in a high-dimensional space. The result captures *semantic relationships*: words that play similar roles in the conversation — even if they are not synonyms — cluster together.

**How it works:** The model uses a sliding window (size 6) that moves across each segment. For every word, the model learns to predict its surrounding words. Through thousands of iterations across all 569 segments, word vectors are adjusted until words with similar contexts have similar vectors. Each word ends up as a 100-dimensional vector. Words like *anxiety* and *stress* will have similar vectors not because they are synonymous but because they appear in similar conversational contexts (*"I feel a lot of [anxiety/stress] when..."*).

**Why it was chosen:** Unlike a word frequency count or TF-IDF, Word2Vec captures *meaning in context*. It can reveal that the group uses *partner* and *boyfriend/girlfriend* interchangeably, or that *isolation* and *withdraw* form a semantic cluster even though they would appear as separate entries in a frequency list. This is particularly valuable for focus group data where the same concept is often expressed in diverse ways. Training on the corpus itself (rather than a generic pretrained model) ensures the vectors reflect the specific vocabulary and conversational patterns of *this* group.

**Model parameters:**
- `vector_size=100` — each word represented as a 100-dimensional vector
- `window=6` — considers 6 words of context on either side
- `min_count=4` — only words appearing at least 4 times across all meetings are included
- `epochs=150` — number of full training passes; higher values improve quality on small corpora

---

### 7. PCA (Principal Component Analysis) for Visualisation

**What it does:** PCA reduces the 100-dimensional Word2Vec vectors to 2 dimensions so they can be plotted on a standard x/y graph.

**How it works:** PCA finds the two directions in the 100-dimensional space that capture the most variance in the data — the two axes along which words vary the most from each other — and projects all word vectors onto those axes. Some information is necessarily lost; the axes of the 2D plot explain a portion of the total variance, reported on the axis labels.

**Why it was chosen:** PCA is the standard first step for visualising high-dimensional embeddings in research contexts. It is linear and deterministic, making the output reproducible. It is interpretable: the distance between two words on the plot reflects their semantic distance in the embedding space, though with some distortion from the dimensionality reduction.

**Output:**
- `themes_fig4_word_clusters.png` — 2D scatter plot of words, colored by semantic cluster

---

### 8. K-Means Clustering

**What it does:** K-Means groups the Word2Vec word vectors into 7 clusters based on geometric proximity in the 100-dimensional embedding space, before PCA reduction.

**How it works:** The algorithm places 7 cluster centres at random positions and iteratively reassigns each word to its nearest centre, then moves each centre to the average position of its assigned words. This repeats until the assignments stabilise. The result is 7 groups of words that are semantically close to each other within the embedding space.

**Why it was chosen:** K-Means is applied *before* PCA (on the full 100-dimensional vectors) so clustering is based on the richest possible representation of meaning, not the compressed 2D version. The clusters provide the color-coding in the word map, making semantic groupings visually immediate. The number of clusters is set to 7 to match the number of LDA topics, providing a consistent thematic frame across both analyses.

**Output:**
- Color groups in `themes_fig4_word_clusters.png`

---

## Output Files Summary

### CSVs

| File | Rows | Description |
|------|------|-------------|
| `sentiment_analysis_output.csv` | 102 | Segment-level analysis for the single pilot transcript |
| `all_meetings_summary.csv` | 6 | One row per meeting; all sentiment metrics, valence/intensity counts, and emotion proportions/counts |
| `all_meetings_segments.csv` | 627 | One row per speaking turn across all six meetings; full segment-level scores |

### Graphs — Per Meeting (6 files)

Each file named `M[N]_[Topic]_summary.png` contains three panels:
- **A** — Valence distribution (positive / neutral / negative counts)
- **B** — NRC emotion profile (average proportion of each Plutchik emotion)
- **C** — Emotional arc (VADER compound score across the meeting, with smoothed trend line)

### Graphs — Cross-Meeting (3 files)

| File | Description |
|------|-------------|
| `CM1_sentiment_comparison.png` | Four key metrics compared across all meetings: mean compound score, % positive turns, % negative turns, mean subjectivity |
| `CM2_emotion_profiles.png` | Left: grouped bar of emotion intensity per meeting; Right: stacked bar of dominant emotion composition per meeting |
| `CM3_emotional_arcs_overlaid.png` | All six emotional arcs overlaid on one chart, x-axis normalised to 0–100% so different-length meetings are comparable |

### Graphs — Thematic (5 files)

| File | Description |
|------|-------------|
| `themes_fig1_distinctive_keywords.png` | Top 12 TF-IDF words most unique to each meeting |
| `themes_fig2_shared_keyword_heatmap.png` | TF-IDF scores for top keywords across all meetings |
| `themes_fig3_topic_model.png` | LDA: 12 defining words per theme + meeting-level theme weights |
| `themes_fig4_word_clusters.png` | Word2Vec 2D semantic cluster map |
| `themes_fig5_topic_wordclouds.png` | Word cloud for each of the 7 discovered themes |

---

## How to Read the Theme Graphs

### Figure 1 — Distinctive Keywords Per Meeting (`themes_fig1_distinctive_keywords.png`)

The figure contains six bar charts arranged in a 2×3 grid — one per meeting. Each chart shows the 12 words that are most *distinctive* to that meeting relative to all others.

**Reading the bars:**
- The x-axis is the TF-IDF score. A longer bar means the word is both frequently used in *this* meeting and rarely used in the others.
- The bars within a chart are ranked from most to least distinctive, with the most distinctive word at the top.
- Darker shading within a bar indicates a higher score.

**What to look for:**
- Words near the top of a chart are that meeting's "signature vocabulary" — the concepts that set it apart from the other sessions.
- If a word appears in the distinctive keyword chart for multiple meetings, it is a shared preoccupation expressed in similar language across sessions. (The heatmap in Figure 2 makes this comparison easier.)
- Short bars near the bottom of a chart indicate words that are only slightly more distinctive than average — treat them as weaker signals.

**What this graph cannot tell you:** TF-IDF only measures distinctiveness, not emotional valence or how the word is being used. A high-scoring word like *conflict* could appear in a meeting because participants were discussing, resolving, or avoiding conflict — the score alone does not distinguish these. Always return to the source transcript to interpret context.

---

### Figure 2 — Shared Keyword Heatmap (`themes_fig2_shared_keyword_heatmap.png`)

This is a matrix where each row is a meeting and each column is a keyword. The colour of each cell shows the TF-IDF score for that word in that meeting.

**Reading the colour:**
- **Darker/more saturated cells** = that word is highly distinctive for that meeting.
- **Light or white cells** = the word is absent or unremarkable in that meeting.

**Reading the column order:**
- Keywords are sorted left to right from most to least *shared* — words on the far left appear in the most meetings; words on the far right are specific to one or two sessions.
- A column that is uniformly dark across all rows represents a theme that runs through the entire conversation series.
- A column that is dark in only one or two rows represents a session-specific preoccupation.

**What to look for:**
- **Horizontal bands of colour** (an entire row is darker than others): one meeting had unusually distinctive or concentrated vocabulary — it may have covered different ground from the others.
- **Vertical bands of colour** (a column is dark across most rows): a recurring concept that participants returned to across multiple sessions regardless of the stated meeting topic.
- **Isolated dark cells**: a word that was highly prominent in exactly one meeting but irrelevant elsewhere — often corresponds to a unique event, topic, or group dynamic in that session.

---

### Figure 3 — LDA Topic Model (`themes_fig3_topic_model.png`)

This figure has two panels side by side.

**Left panel — Defining words per theme:**
There is one horizontal bar chart per discovered theme (Theme 1 through Theme 7), stacked vertically. Each bar chart shows the 12 words with the highest probability of belonging to that theme.

- The x-axis is the word's *probability within that theme* — how characteristic it is of this theme relative to all other words.
- The bars are ranked most-to-least characteristic, with the strongest word at the top.
- The colour is consistent per theme and matches the colour used in the right panel.
- The theme title (e.g., "Theme 3: Relationship") is automatically generated from the single most content-rich word in that theme and is a starting point for interpretation, not a definitive label. The researcher should read all 12 words together to decide what the theme represents.

**Right panel — Theme presence per meeting (heatmap):**
Rows are themes; columns are meetings. Each cell shows the average weight of that theme in that meeting, where weight reflects how much of the speaking turns in that meeting were "about" that theme according to the model.

- **Darker blue cells** = that theme was more prominent in that meeting.
- **Each column** shows the thematic profile of one meeting — which themes dominated and which were absent.
- **Each row** shows which meetings most expressed a given theme.

**What to look for:**
- A theme with a uniformly high weight across all meetings is a *cross-cutting theme* — something participants returned to regardless of the session's stated focus.
- A theme with high weight in only one or two meetings is a *session-specific theme* — it may reflect the unique framing or prompts of those sessions.
- Two themes with similar patterns across meetings (both high in the same meetings, both low in the same meetings) may be related or overlapping constructs — read their top words together to assess.

**Important limitation:** The left and right panels are linked — a theme's label comes from its top words (left), and its distribution is shown on the right. If the top words for a theme feel incoherent, the model may have struggled to find a clean separation for that theme. This does not mean the other themes are invalid; LDA topics vary in their interpretability.

---

### Figure 4 — Word Embedding Cluster Map (`themes_fig4_word_clusters.png`)

This is a 2D scatter plot where every point is a word and its position reflects how the model understands its meaning based on conversational context across all six meetings.

**Reading position:**
- **Words that appear close together** were consistently used in similar conversational contexts — they belong to the same semantic neighbourhood. This does not mean they are synonyms; it means the conversations treated them similarly (e.g., *anxiety* and *stress* may cluster together because participants used them in structurally similar sentences).
- **Words that appear far apart** were used in very different contexts and rarely co-occurred.
- The x- and y-axes are mathematical (PCA components) and have no inherent psychological meaning. Only the *relative distance between words* should be interpreted, not their absolute position.

**Reading colour:**
- Each colour represents one of the seven K-Means clusters — groups of words that are geometrically close to each other in the full 100-dimensional embedding space (before compression to 2D).
- Words of the same colour form a semantic cluster. Together, they describe a coherent conversational domain or concept area.
- The dashed outlines (convex hulls) show the rough boundary of each cluster to help visually separate overlapping regions.

**Reading font weight:**
- **Bold words** appear at least 20 times across all meetings — they are high-frequency terms and anchor the cluster's meaning.
- **Normal weight words** are lower frequency but still semantically part of that cluster.

**What to look for:**
- **Tight, compact clusters** where many related words sit close together suggest a well-defined thematic domain that participants discussed in a consistent way.
- **Loose or scattered clusters** suggest a broader or more diffuse topic — participants used the associated words in varied contexts.
- **Words near the boundary between two clusters** are semantically bridge concepts — they appear in contexts that straddle two themes (e.g., a word like *support* might sit between a mental health cluster and a relationships cluster).
- **Outlier words** positioned far from any cluster may represent unique or idiosyncratic uses of language in this corpus.

**Comparison with Figure 3:** The clusters here are derived independently from the LDA topics. However, if a cluster and an LDA theme share several of the same words, that convergence across two separate methods strengthens confidence that the theme is a genuine, stable pattern in the data.

---

### Figure 5 — Topic Word Clouds (`themes_fig5_topic_wordclouds.png`)

There is one word cloud for each of the seven LDA themes.

**Reading a word cloud:**
- **Word size** reflects the word's probability within that theme — larger words are more strongly associated with the theme than smaller ones.
- **Word colour** varies within each cloud but all shades are drawn from the same base colour as that theme (to match Figure 3).
- Each cloud is independent — a word's size in one cloud cannot be directly compared to the same word's size in another cloud, because the scales are set per-theme.

**What word clouds are good for:**
- Getting an immediate visual sense of a theme's character — the largest words name the core concepts.
- Quickly communicating themes to audiences unfamiliar with the underlying method.
- Identifying whether the top words cluster around a single coherent concept or reflect a theme that mixes several ideas.

**What word clouds are not good for:**
- Precise quantitative comparison. Because each cloud is independently scaled, a word appearing large in one cloud and small in another does not necessarily mean it is more important in the first theme — it just means it is more dominant relative to the *other words in that theme*.
- Words that appear in multiple clouds are shared across themes — this overlap is real and should be noted rather than treated as an error.

**Practical use:** Word clouds work well as a starting point for researcher-led theme naming. Reading the top five or six words per cloud, consulting the corresponding bar chart in Figure 3 for more detail, and then returning to the transcript excerpts is the recommended workflow for arriving at a theoretically grounded theme label.

---

## Key Methodological Decisions

**Why paragraph-level, not sentence-level analysis?**
Focus group speech frequently contains run-on sentences, sentence fragments, and mid-thought corrections. Sentence boundaries are unreliable in spoken transcripts. Paragraph-level segmentation (by blank line) better reflects natural speaking turns and provides enough text per unit for reliable sentiment scoring.

**Why multiple sentiment models?**
No single model captures the full picture. VADER is fast and validated for informal speech but only measures positive/negative valence. TextBlob adds subjectivity. NRC adds emotion category (qualitative type of emotion, not just direction). Using all three in parallel provides a richer, triangulated view — consistent with best practice in mixed-methods psychological research.

**Why unsupervised (LDA, Word2Vec) rather than pre-coded themes?**
Pre-coding themes requires a theoretical framework specified in advance. LDA and Word2Vec are *exploratory* — they let the content of these specific conversations determine the thematic structure. This is particularly appropriate in early-stage focus group research where the range of themes is not yet known. Unsupervised outputs can then be validated against or used to refine a deductive coding scheme.

---

## Limitations

- **Speaker identity is not tracked.** The transcripts do not consistently label individual speakers, so all analysis is at the group level. Sentiment patterns cannot be attributed to specific participants.
- **Context blindness.** Sentiment models score language at face value. Sarcasm, irony, and clinical descriptions of past distress may be misclassified. All automated scores should be reviewed against the source excerpts.
- **Small corpus.** Word2Vec performs best on large corpora (millions of words). With ~55,000 total words across six meetings, the word vectors are functional but less reliable than those trained on larger datasets. The semantic cluster map should be interpreted directionally, not as definitive.
- **LDA topic number is a hyperparameter.** Setting `n_topics=7` was a researcher judgment call. Different values will yield different theme structures. The model makes no claim that there are "exactly 7 themes."

---

## References

- Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *ICWSM*.
- Mohammad, S. M., & Turney, P. D. (2013). Crowdsourcing a word-emotion association lexicon. *Computational Intelligence, 29*(3), 436–465.
- Mikolov, T., et al. (2013). Efficient estimation of word representations in vector space. *arXiv:1301.3781*.
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. *Journal of Machine Learning Research, 3*, 993–1022.
- Plutchik, R. (1980). *Emotion: A psychoevolutionary synthesis*. Harper & Row.
