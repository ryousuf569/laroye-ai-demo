import json
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def compute_semantic_similarity(embedder, intentions, title, description):
    """Compute cosine similarity between parent intentions and content text.

    Kept for transparency: the score is returned in the output so callers can
    inspect it, but it is no longer used to override the classifier's prediction.
    Empirical testing showed that cosine similarity between short intent keywords
    and full content sentences has near-identical distributions across all three
    label classes (strong/partial/weak mean ~0.26–0.29) — it provides no reliable
    discriminative signal and was incorrectly suppressing valid strong_match
    predictions.
    """
    intent_vec = embedder.encode([intentions])
    content_vec = embedder.encode([f"{title} {description}"])
    sim = cosine_similarity(intent_vec, content_vec)[0][0]
    return float(max(0.0, min(1.0, sim)))


# predict on new content
def predict_match(age, intentions, title, description, saved_model=None, embedder=None):
    """Given a piece of content, predict how well it matches (+ probabilities).

    saved_model and embedder can be passed in from a cached startup load (e.g.
    in app.py) to avoid reloading the 80MB model on every request.  If omitted,
    they are loaded from disk — useful for one-off testing from the command line.
    """

    if saved_model is None or embedder is None:
        # fallback: load from disk (slow — avoid in production / live demos)
        with open(r"..\artifacts\model.pkl", "rb") as f:
            saved_model = pickle.load(f)
        with open(r"..\artifacts\metadata.json", "r") as f:
            metadata = json.load(f)
        embedder = SentenceTransformer(metadata["embed_model"])

    text = f"age: {age} | intentions: {intentions} | title: {title} | description: {description}"
    vec = embedder.encode([text])

    probs = saved_model.predict_proba(vec)[0]

    # pair each class name with its probability
    prob_dict = dict(zip(saved_model.classes_, probs))

    # semantic similarity is computed and returned for transparency / debugging
    # but does NOT override the classifier — see docstring above for why
    semantic_sim = compute_semantic_similarity(embedder, intentions, title, description)

    # derive the final label and a weighted match score from classifier probabilities
    label = max(prob_dict, key=prob_dict.get)
    match_score = (prob_dict.get("strong_match", 0.0) * 100
                   + prob_dict.get("partial_match", 0.0) * 60
                   + prob_dict.get("weak_match", 0.0) * 20) / 100

    return {
        "label": label,
        "probabilities": {k: round(float(v), 4) for k, v in prob_dict.items()},
        "match_score": round(match_score, 4),
        "semantic_similarity": round(semantic_sim, 4),
    }
