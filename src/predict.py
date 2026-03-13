import json
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def compute_semantic_similarity(embedder, intentions, title, description):
    """Compute cosine similarity between parent intentions and content text."""
    intent_vec = embedder.encode([intentions])
    content_vec = embedder.encode([f"{title} {description}"])
    sim = cosine_similarity(intent_vec, content_vec)[0][0]
    # clamp to [0, 1] since intentions/content are always positive-sense text
    return float(max(0.0, min(1.0, sim)))


def adjust_probabilities(prob_dict, semantic_sim):
    """Adjust class probabilities based on semantic similarity.

    Low similarity  -> penalize strong_match, boost partial/weak
    High similarity -> keep strong_match as-is
    """
    adjusted = dict(prob_dict)

    # penalty grows as similarity drops below 0.5
    # at sim=0.5 penalty=0, at sim=0 penalty=0.5
    penalty = max(0.0, 0.5 - semantic_sim)

    # reduce strong_match probability
    if "strong_match" in adjusted:
        adjusted["strong_match"] = max(0.0, adjusted["strong_match"] - penalty)

    # give a small boost to partial_match when similarity is moderate (0.25-0.55)
    if "partial_match" in adjusted and 0.25 <= semantic_sim <= 0.55:
        adjusted["partial_match"] += penalty * 0.5

    # re-normalize so probabilities sum to 1
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}

    return adjusted

def apply_similarity_label_ceiling(prob_dict, semantic_sim):
    """Prevent topic-misaligned content from scoring too highly.

    Very low similarity  -> cannot be above weak_match
    Low/moderate similarity -> cannot be above partial_match
    High similarity -> no ceiling
    """
    adjusted = dict(prob_dict)

    if semantic_sim < 0.28:
        # Only weak_match allowed
        adjusted["strong_match"] = 0.0
        adjusted["partial_match"] = 0.0

    elif semantic_sim < 0.45:
        # strong_match not allowed, but partial_match is okay
        adjusted["strong_match"] = 0.0

    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}

    return adjusted

# predict on new content
def predict_match(age, intentions, title, description):
    """Given a piece of content, predict how well it matches (+ probabilities)."""

    # load saved artifacts
    with open(r"..\artifacts\model.pkl", "rb") as f:
        saved_model = pickle.load(f)
    with open(r"..\artifacts\metadata.json", "r") as f:
        metadata = json.load(f)

    # encode using the same sentence-transformer model from training
    embedder = SentenceTransformer(metadata["embed_model"])
    text = f"age: {age} | intentions: {intentions} | title: {title} | description: {description}"
    vec = embedder.encode([text])

    probs = saved_model.predict_proba(vec)[0]

    # pair each class name with its probability
    prob_dict = dict(zip(saved_model.classes_, probs))

    # semantic alignment: compare intentions to content directly
    semantic_sim = compute_semantic_similarity(embedder, intentions, f"{title}", description)

    # adjust probabilities using semantic similarity
    prob_dict = adjust_probabilities(prob_dict, semantic_sim)

    #adjust labels for final verdict
    prob_dict = apply_similarity_label_ceiling(prob_dict, semantic_sim)

    # recompute label and match score from adjusted probabilities
    label = max(prob_dict, key=prob_dict.get)
    match_score = (prob_dict.get("strong_match", 0.0) * 100 
                   + prob_dict.get("partial_match", 0.0) * 60
                   + prob_dict.get("weak_match", 0.0) * 20) / 100

    return {
        "label": label,
        "probabilities": prob_dict,
        "match_score": round(match_score, 4),
        "semantic_similarity": round(semantic_sim, 4),
    }
