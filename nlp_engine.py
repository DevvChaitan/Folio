"""Text preprocessing and semantic embeddings for Folio profiles.

The Sentence Transformer is a pretrained feature extractor. The feedback model
learns matching weights separately; this module never trains transformer weights.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Iterable

import numpy as np


SENTENCE_TRANSFORMER_NAME = "all-MiniLM-L6-v2"

_MODEL = None
_WORDNET_LEMMATIZER = None
_WORDNET_CHECKED = False

# Keep negation terms: removing "not" or "never" can reverse profile meaning.
_NEGATIONS = {"no", "not", "nor", "never", "without"}
_STOP_WORDS = frozenset(
    "a an the and or but if then than so because as at by for from in into of "
    "on onto to with about above after again against all am are is was were be "
    "been being below between both can could did do does doing down during each "
    "few further had has have having he her here hers herself him himself his "
    "how i me my myself it its itself just more most must myself neither once "
    "only other our ours ourselves out over own same she should some such than "
    "their theirs them themselves there these they this those through too under "
    "until up very we what when where which while who whom why will would you "
    "your yours yourself yourselves also really much many often ever every"
    .split()
)

_IRREGULAR_LEMMAS = {
    "am": "be",
    "are": "be",
    "is": "be",
    "was": "be",
    "were": "be",
    "been": "be",
    "being": "be",
    "has": "have",
    "had": "have",
    "does": "do",
    "did": "do",
    "done": "do",
    "went": "go",
    "gone": "go",
    "better": "good",
    "best": "good",
    "worse": "bad",
    "worst": "bad",
    "people": "person",
    "children": "child",
    "men": "man",
    "women": "woman",
}
_FALLBACK_E_STEMS = {
    "bik",
    "chang",
    "clos",
    "creat",
    "cycl",
    "danc",
    "driv",
    "giv",
    "lov",
    "mak",
    "mov",
    "not",
    "produc",
    "shar",
    "tak",
    "us",
    "wast",
    "writ",
}


def _fallback_lemma(token: str) -> str:
    """Reduce common English inflections when NLTK WordNet is unavailable."""
    if token in _IRREGULAR_LEMMAS:
        return _IRREGULAR_LEMMAS[token]

    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"

    if len(token) > 5 and token.endswith("ing"):
        root = token[:-3]
        if len(root) > 2 and root[-1] == root[-2]:
            root = root[:-1]
        if root.endswith(("at", "iz")) or root in _FALLBACK_E_STEMS:
            root += "e"
        return root

    if len(token) > 4 and token.endswith("ed"):
        root = token[:-2]
        if len(root) > 2 and root[-1] == root[-2]:
            root = root[:-1]
        if root.endswith(("at", "iz")) or root in _FALLBACK_E_STEMS:
            root += "e"
        return root

    if len(token) > 5 and token.endswith("sses"):
        return token[:-2]

    if len(token) > 4 and token.endswith(("ches", "shes", "xes", "zes")):
        return token[:-2]

    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]

    return token


def _lemmatize(token: str) -> str:
    """Use WordNet when installed, with a deterministic local fallback."""
    global _WORDNET_LEMMATIZER, _WORDNET_CHECKED

    if not _WORDNET_CHECKED:
        _WORDNET_CHECKED = True
        try:
            from nltk.stem import WordNetLemmatizer

            candidate = WordNetLemmatizer()
            # This also checks whether the WordNet corpus is installed locally.
            candidate.lemmatize("profiles", pos="n")
            _WORDNET_LEMMATIZER = candidate
        except (ImportError, LookupError):
            _WORDNET_LEMMATIZER = None

    if _WORDNET_LEMMATIZER is not None:
        try:
            verb_lemma = _WORDNET_LEMMATIZER.lemmatize(token, pos="v")
            return _WORDNET_LEMMATIZER.lemmatize(verb_lemma, pos="n")
        except LookupError:
            _WORDNET_LEMMATIZER = None

    return _fallback_lemma(token)


def preprocess_text(text: object, *, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """Normalize profile text before embedding it.

    Punctuation and markup are removed, common stopwords are dropped while
    negations are retained, and words are lemmatized when a local WordNet corpus
    is available. No corpus or model is downloaded by this preprocessing step.
    """
    if text is None:
        return ""

    normalized = unicodedata.normalize("NFKC", html.unescape(str(text))).casefold()
    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?", normalized)
    cleaned = []
    for token in tokens:
        if remove_stopwords and token in _STOP_WORDS and token not in _NEGATIONS:
            continue
        cleaned.append(_lemmatize(token) if lemmatize else token)
    return " ".join(cleaned)


def _get_sentence_transformer():
    """Load the pretrained text encoder only when an embedding is requested."""
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Sentence Transformers is required for Folio text embeddings. "
                "Install the project dependencies in the active Python environment."
            ) from exc
        _MODEL = SentenceTransformer(SENTENCE_TRANSFORMER_NAME)
    return _MODEL


def get_text_embeddings(
    text_list: Iterable[object],
    *,
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """Return normalized embeddings from the pretrained Sentence Transformer."""
    if isinstance(text_list, (str, bytes)):
        raise TypeError("text_list must be an iterable of profile texts, not one string")
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    texts = list(text_list)
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    cleaned_texts = [preprocess_text(text) for text in texts]
    encoder = _get_sentence_transformer()
    dimension = encoder.get_sentence_embedding_dimension()
    if not dimension:
        raise RuntimeError("The configured Sentence Transformer has no known output dimension")

    embeddings = np.zeros((len(cleaned_texts), dimension), dtype=np.float32)
    nonempty_indexes = [i for i, text in enumerate(cleaned_texts) if text]
    if nonempty_indexes:
        encoded = encoder.encode(
            [cleaned_texts[i] for i in nonempty_indexes],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )
        embeddings[nonempty_indexes] = np.asarray(encoded, dtype=np.float32)
    return embeddings


def calculate_semantic_similarity(vec1: object, vec2: object) -> float:
    """Return cosine similarity on a 0–1 scale; empty vectors score zero."""
    first = np.asarray(vec1, dtype=np.float32).reshape(-1)
    second = np.asarray(vec2, dtype=np.float32).reshape(-1)
    if first.size == 0 or first.shape != second.shape:
        raise ValueError("Both embeddings must be non-empty vectors with the same dimensions")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Embeddings must contain only finite values")

    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0

    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return float(np.clip(cosine, 0.0, 1.0))


if __name__ == "__main__":
    example_bios = [
        "Data analyst focused on predictive modeling and structured data pipelines.",
        "Machine learning engineer building classification algorithms and analytics tools.",
        "Creative director specializing in graphic design and frontend aesthetics.",
    ]
    example_embeddings = get_text_embeddings(example_bios, show_progress_bar=True)
    print(
        "Similarity (Data vs ML): "
        f"{calculate_semantic_similarity(example_embeddings[0], example_embeddings[1]):.4f}"
    )
    print(
        "Similarity (Data vs Design): "
        f"{calculate_semantic_similarity(example_embeddings[0], example_embeddings[2]):.4f}"
    )
