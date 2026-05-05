import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
 
from embeddings_lab import load_glove, extract_bert_embedding
from transformers import AutoTokenizer, AutoModel
 
# ── style ─────────────────────────────────────────────────────────────────────
 
BG_COLOR     = "#ffffff"
PANEL_BG     = "#f8f9fa"
GRID_COLOR   = "#dee2e6"
TEXT_COLOR   = "#212529"
ACCENT_COLOR = "#000000"
 
plt.rcParams.update({
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "axes.titlecolor":   TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    0.6,
    "legend.facecolor":  BG_COLOR,
    "legend.edgecolor":  GRID_COLOR,
    "legend.labelcolor": TEXT_COLOR,
    "text.color":        TEXT_COLOR,
    "font.family":       "DejaVu Sans",
})
 
# ── word categories ───────────────────────────────────────────────────────────
 
WORD_CATEGORIES_SEED = {
    "sports": [
        "football", "basketball", "tennis", "cricket", "rugby",
        "swimming", "athletics", "cycling", "boxing", "golf",
        "volleyball", "hockey", "baseball", "skiing", "rowing",
        "championship", "tournament", "stadium", "referee", "coach",
        "goalkeeper", "midfielder", "striker", "athlete", "olympic",
        "medal", "trophy", "league", "fixture", "transfer",
        "penalty", "tackle", "dribble", "sprint", "marathon",
        "trainer", "squad", "captain", "substitute", "offside"
    ],
    "politics": [
        "parliament", "government", "election", "democracy", "president",
        "minister", "senate", "congress", "policy", "legislation",
        "referendum", "constitution", "ambassador", "diplomat", "coalition",
        "opposition", "campaign", "vote", "ballot", "party",
        "conservative", "liberal", "socialist", "republican", "democrat",
        "chancellor", "cabinet", "debate", "reform", "treaty",
        "sanction", "veto", "jurisdiction", "sovereignty", "taxation",
        "corruption", "lobbying", "manifesto", "impeachment", "bureaucracy"
    ],
    "emotions": [
        "happy", "sad", "angry", "fearful", "surprised",
        "disgusted", "anxious", "excited", "nervous", "calm",
        "depressed", "joyful", "frustrated", "hopeful", "lonely",
        "proud", "ashamed", "jealous", "grateful", "confused",
        "love", "hate", "grief", "rage", "panic",
        "relief", "guilt", "envy", "trust", "despair",
        "enthusiasm", "melancholy", "nostalgia", "euphoria", "empathy",
        "sympathy", "compassion", "resentment", "boredom", "curiosity"
    ],
    "technology": [
        "computer", "software", "internet", "algorithm", "database",
        "network", "processor", "smartphone", "wireless", "digital",
        "encryption", "bandwidth", "semiconductor", "browser", "server",
        "firewall", "compiler", "interface", "pixel", "binary",
        "cloud", "streaming", "bluetooth", "satellite", "router",
        "hardware", "keyboard", "monitor", "printer", "scanner",
        "microchip", "voltage", "transistor", "antenna", "protocol",
        "cybersecurity", "artificial", "machine", "robot", "automation"
    ],
    "countries": [
        "france", "germany", "japan", "brazil", "australia",
        "canada", "india", "china", "russia", "italy",
        "spain", "mexico", "argentina", "egypt", "nigeria",
        "kenya", "sweden", "norway", "denmark", "finland",
        "poland", "ukraine", "turkey", "iran", "iraq",
        "pakistan", "indonesia", "thailand", "vietnam", "malaysia",
        "colombia", "chile", "peru", "venezuela", "portugal",
        "greece", "hungary", "romania", "belgium", "netherlands"
    ]
}
 
WORD_COLORS = {
    "sports":     "#e63946",
    "politics":   "#1d7cb4",
    "emotions":   "#2a9d8f",
    "technology": "#e76f51",
    "countries":  "#7b2d8b"
}
 
DOC_COLORS = {
    "business":      "#e63946",
    "entertainment": "#1d7cb4",
    "politics":      "#2a9d8f",
    "sport":         "#e76f51",
    "tech":          "#7b2d8b"
}
 
ANNOTATE_WORDS = {
    "sports":     ["olympic", "marathon", "stadium", "championship"],
    "politics":   ["democracy", "election", "parliament", "corruption"],
    "emotions":   ["love", "panic", "euphoria", "melancholy"],
    "technology": ["algorithm", "semiconductor", "robot", "machine"],
    "countries":  ["japan", "brazil", "nigeria", "ukraine"]
}
 
 
# ── helpers ───────────────────────────────────────────────────────────────────
 
def select_words(glove, seed):
    words, vectors, categories = [], [], []
    for category, word_list in seed.items():
        found   = [w for w in word_list if w in glove]
        missing = [w for w in word_list if w not in glove]
        if missing:
            print(f"  {category}: skipped OOV — {missing}")
        for word in found:
            words.append(word)
            vectors.append(glove[word])
            categories.append(category)
        print(f"  {category}: {len(found)}/{len(word_list)} words found")
    return words, np.array(vectors), categories
 
 
def find_boundary_words(words, vectors, categories, top_n=3):
    category_list = list(WORD_CATEGORIES_SEED.keys())
    centroids = {
        cat: vectors[[i for i, c in enumerate(categories) if c == cat]].mean(axis=0)
        for cat in category_list
    }
    centroid_matrix = np.array([centroids[c] for c in category_list])
 
    boundary = []
    for i, (word, cat) in enumerate(zip(words, categories)):
        sims        = cosine_similarity(vectors[i].reshape(1, -1), centroid_matrix).flatten()
        nearest_idx = int(np.argmax(sims))
        nearest_cat = category_list[nearest_idx]
        true_idx    = category_list.index(cat)
        if nearest_cat != cat:
            boundary.append({
                "word":             word,
                "true_category":    cat,
                "nearest_category": nearest_cat,
                "similarity_gap":   float(sims[nearest_idx] - sims[true_idx])
            })
 
    boundary.sort(key=lambda x: x["similarity_gap"], reverse=True)
    return boundary[:top_n]
 
 
def _annotate_word(ax, word, x, y, bold=False, italic=False):
    """Draw a word label with a dark shadow for readability on dark background."""
    style = {}
    if bold:
        style["fontweight"] = "bold"
    if italic:
        style["fontstyle"] = "italic"
    ax.annotate(
        word, (x, y),
        fontsize=7.5,
        color=TEXT_COLOR,
        xytext=(6, 5),
        textcoords="offset points",
        path_effects=[
            pe.withStroke(linewidth=2, foreground=BG_COLOR)
        ],
        **style
    )
 
 
def _scatter_word_ax(ax, vectors_2d, words, categories, boundary_words, title):
    """Render a single word embedding scatter on the given axes."""
    annotate_set  = {w for ws in ANNOTATE_WORDS.values() for w in ws}
    boundary_set  = {b["word"] for b in boundary_words}
 
    ax.grid(True, alpha=0.3)
 
    for category in WORD_CATEGORIES_SEED:
        mask = [i for i, c in enumerate(categories) if c == category]
        ax.scatter(
            vectors_2d[mask, 0], vectors_2d[mask, 1],
            c=WORD_COLORS[category],
            label=category,
            alpha=0.85,
            s=55,
            edgecolors="none"
        )
 
    # annotate notable words
    for i, word in enumerate(words):
        if word in annotate_set:
            _annotate_word(ax, word, vectors_2d[i, 0], vectors_2d[i, 1], bold=True)
 
    # mark and annotate boundary words
    for i, word in enumerate(words):
        if word in boundary_set:
            ax.scatter(
                vectors_2d[i, 0], vectors_2d[i, 1],
                s=160,
                facecolors="none",
                edgecolors=ACCENT_COLOR,
                linewidths=2.0,
                zorder=6
            )
            _annotate_word(ax, f"* {word}", vectors_2d[i, 0], vectors_2d[i, 1],
                           italic=True)
 
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Dimension 1", fontsize=9, labelpad=8)
    ax.set_ylabel("Dimension 2", fontsize=9, labelpad=8)
    legend = ax.legend(
        title="Category", fontsize=8, title_fontsize=9,
        loc="best", framealpha=0.4, markerscale=1.3
    )
 
 
# ── plot functions ────────────────────────────────────────────────────────────
 
def plot_tsne_standalone(words, tsne_2d, categories, boundary_words):
    """Standalone t-SNE word plot — assignment deliverable."""
    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor(BG_COLOR)
 
    _scatter_word_ax(
        ax, tsne_2d, words, categories, boundary_words,
        "GloVe Word Embeddings — t-SNE 2D Projection  "
        "(* = boundary word)"
    )
 
    fig.suptitle(
        "5 Semantic Categories · 199 Words · perplexity=30",
        fontsize=10, color=TEXT_COLOR, y=0.97, alpha=0.7
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("word_embeddings.png", dpi=150,
                bbox_inches="tight", facecolor=BG_COLOR)
    print("Saved word_embeddings.png")
    plt.close()
 
 
def plot_comparison(words, tsne_2d, pca_2d, categories, boundary_words,
                    pca_variance):
    """Side-by-side t-SNE vs PCA — enhancement deliverable."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
    fig.patch.set_facecolor(BG_COLOR)
 
    _scatter_word_ax(
        ax1, tsne_2d, words, categories, boundary_words,
        "t-SNE  ·  preserves local structure  ·  perplexity=30"
    )
    _scatter_word_ax(
        ax2, pca_2d, words, categories, boundary_words,
        f"PCA  ·  preserves global structure  ·  "
        f"variance explained {pca_variance:.1%}"
    )
 
    fig.suptitle(
        "GloVe Word Embeddings — t-SNE vs PCA\n"
        "(*) boundary words are closer to a different category centroid in 50d space",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, y=1.01
    )
    plt.tight_layout()
    plt.savefig("word_embeddings_comparison.png", dpi=150,
                bbox_inches="tight", facecolor=BG_COLOR)
    print("Saved word_embeddings_comparison.png")
    plt.close()
 
 
def plot_document_embeddings(labels, vectors_2d, categories, explained_variance):
    """Standalone document PCA plot — assignment deliverable."""
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(BG_COLOR)
 
    ax.grid(True, alpha=0.3)
 
    for category in DOC_COLORS:
        mask = [i for i, c in enumerate(categories) if c == category]
        if not mask:
            continue
        ax.scatter(
            vectors_2d[mask, 0], vectors_2d[mask, 1],
            c=DOC_COLORS[category],
            label=category,
            alpha=0.9,
            s=130,
            edgecolors="none"
        )
 
    for i, label in enumerate(labels):
        ax.annotate(
            label[:30],
            (vectors_2d[i, 0], vectors_2d[i, 1]),
            fontsize=7.5,
            color=TEXT_COLOR,
            xytext=(6, 5),
            textcoords="offset points",
            path_effects=[pe.withStroke(linewidth=2, foreground=BG_COLOR)]
        )
 
    ax.set_title(
        f"DistilBERT Document Embeddings — PCA 2D Projection\n"
        f"PC1={explained_variance[0]:.1%}  ·  "
        f"PC2={explained_variance[1]:.1%}  ·  "
        f"Total={sum(explained_variance):.1%}",
        fontsize=12, fontweight="bold", pad=14
    )
    ax.set_xlabel("PCA Component 1", fontsize=10, labelpad=8)
    ax.set_ylabel("PCA Component 2", fontsize=10, labelpad=8)
    ax.legend(title="Category", fontsize=9, title_fontsize=10,
              loc="best", framealpha=0.4, markerscale=1.3)
 
    plt.tight_layout()
    plt.savefig("doc_embeddings.png", dpi=150,
                bbox_inches="tight", facecolor=BG_COLOR)
    print("Saved doc_embeddings.png")
    plt.close()
 
 
# ── main ──────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    df   = pd.read_csv("data/bbc_news.csv")
    glove = load_glove("data/glove_50k_50d.txt")
    print(f"Loaded {len(glove)} GloVe vectors")
 
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model     = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()
 
    # ── Part 1: word embeddings ───────────────────────────────────────────────
    print("\nFiltering seed words against GloVe vocabulary...")
    words, word_vectors, word_categories = select_words(glove, WORD_CATEGORIES_SEED)
    print(f"\nTotal words for visualization: {len(words)}")
 
    print("Running t-SNE (50d → 2d)...")
    tsne_2d = TSNE(
        n_components=2, perplexity=30, random_state=42, max_iter=1000
    ).fit_transform(word_vectors)
 
    print("Running PCA (50d → 2d)...")
    pca_words      = PCA(n_components=2, random_state=42)
    pca_word_2d    = pca_words.fit_transform(word_vectors)
    pca_word_var   = pca_words.explained_variance_ratio_.sum()
    print(f"Word PCA explained variance: {pca_word_var:.2%}")
 
    print("\nDetecting boundary words in original 50d space...")
    boundary_words = find_boundary_words(words, word_vectors, word_categories, top_n=3)
    print("Top boundary words:")
    for b in boundary_words:
        print(f"  '{b['word']}' [{b['true_category']}] → "
              f"nearest: [{b['nearest_category']}]  gap={b['similarity_gap']:.4f}")
 
    # three separate plots
    plot_tsne_standalone(words, tsne_2d, word_categories, boundary_words)
    plot_comparison(words, tsne_2d, pca_word_2d, word_categories,
                    boundary_words, pca_word_var)
 
    # ── Part 2: document embeddings ───────────────────────────────────────────
    print("\nSelecting 4 articles per category (20 total)...")
    selected = pd.concat([
        df[df["category"] == cat].head(4)
        for cat in df["category"].unique()
    ]).reset_index(drop=True)
    print(selected["category"].value_counts().to_string())
 
    print("\nComputing DistilBERT embeddings for 20 articles...")
    doc_embeddings = np.array([
        extract_bert_embedding(t, tokenizer, model)
        for t in selected["text"].tolist()
    ])
    print(f"Document embeddings shape: {doc_embeddings.shape}")
 
    print("Running PCA on document embeddings (768d → 2d)...")
    pca_docs = PCA(n_components=2, random_state=42)
    doc_2d   = pca_docs.fit_transform(doc_embeddings)
    print(f"PCA explained variance: {pca_docs.explained_variance_ratio_}")
    print(f"Total variance explained: {pca_docs.explained_variance_ratio_.sum():.2%}")
 
    plot_document_embeddings(
        selected["text"].str[:30].tolist(),
        doc_2d,
        selected["category"].tolist(),
        pca_docs.explained_variance_ratio_
    )