# Laroye Content Matcher Model

A lightweight machine learning system that predicts how well a piece of content matches a parent’s intentions.

The model outputs a probability distribution across three classes:

- **Strong Match**
- **Partial Match**
- **Weak Match**

The goal is to help determine whether short-form content (for example YouTube Shorts) aligns with what a parent wants their child to watch.

---

# Motivation

In research work with **Wat.AI**, I have been working with **FinBERT**, a finance-tuned BERT model designed to analyze financial headlines.

FinBERT produces a probability distribution across three classes:

- Bullish  
- Bearish  
- Neutral  

Instead of forcing a single label, this probabilistic output provides a more interpretable view of model confidence.

Inspired by that approach, the **Content Matcher Model** produces a similar distribution:

```
[strong_match, partial_match, weak_match]
```

This allows downstream systems to reason about **confidence and alignment**, rather than relying on a single prediction.

---

# Model Architecture

The system predicts match quality using the following inputs:

```
age
intentions
title
description
```

The architecture consists of three main stages:

1. **Semantic Encoding**
2. **Logistic Regression Classification**
3. **Semantic Alignment Adjustment**

---

# Baseline Model (TF-IDF + Logistic Regression)

The first version of the system used **TF-IDF vectorization** to convert text into numerical feature vectors.

Pipeline:

```
text → TF-IDF vectorizer → Logistic Regression → probability distribution
```

### Results

```
Accuracy: 93.82%

               precision    recall  f1-score   support

partial_match       0.95      0.95      0.95        59
strong_match        0.92      0.97      0.94        61
weak_match          0.95      0.90      0.92        58

accuracy                                0.94       178
macro avg           0.94      0.94      0.94       178
weighted avg        0.94      0.94      0.94       178
```

### Limitation

Although this model achieved high accuracy, it relied heavily on **token overlap**.

This caused the system to sometimes overpredict **strong_match** for content that simply sounded educational.

Example:

```
Intentions: science, plants
Content: skating safety
```

The model recognized words like *teaches* and *kids*, but failed to detect the **semantic mismatch** between the topic and the parent's intention.

---

# Improved Model (Sentence Transformers + Logistic Regression)

To capture deeper semantic meaning, the model was upgraded to use **Sentence Transformers**.

Instead of counting words, text is converted into **dense semantic embeddings** that capture contextual meaning.

Pipeline:

```
text → Sentence Transformer embeddings → Logistic Regression → probability distribution
```

### Results

```
Accuracy: 82.02%

               precision    recall  f1-score   support

partial_match       0.78      0.75      0.77        68
strong_match        0.81      0.86      0.84        96
weak_match          0.87      0.83      0.85        64

accuracy                                0.82       228
macro avg           0.82      0.81      0.82       228
weighted avg        0.82      0.82      0.82       228
```

While raw accuracy decreased slightly, predictions became **much more semantically meaningful**, which is far more important for real-world use.

The model can now detect relationships such as:

```
"hockey" ≈ "skating"
"plants" ≈ "biology"
"music practice" ≈ "learning instruments"
```

---

# Semantic Alignment Layer

To ensure that content truly aligns with the parent's intent, the prediction pipeline includes a semantic alignment stage.

This layer ensures the model distinguishes between:

- **Educational content**
- **Content aligned with the parent’s stated intentions**

---

# Function: Semantic Similarity

This function computes cosine similarity between the parent's intentions and the content text.

```python
def compute_semantic_similarity(embedder, intentions, title, description):
    """Compute cosine similarity between parent intentions and content text."""
    intent_vec = embedder.encode([intentions])
    content_vec = embedder.encode([f"{title} {description}"])
    sim = cosine_similarity(intent_vec, content_vec)[0][0]

    # clamp similarity to [0,1]
    return float(max(0.0, min(1.0, sim)))
```

This produces a semantic alignment score between the parent’s requested topic and the video content.

---

# Prediction Pipeline

The full prediction process works as follows:

```
Frontend Input
(age, intentions, title, description)

↓
FastAPI backend

↓
Sentence Transformer embeddings

↓
Logistic Regression classifier

↓
Semantic similarity calculation

↓
Final prediction
```

Example output:

```json
{
  "label": "partial_match",
  "probabilities": {
    "strong_match": 0.12,
    "partial_match": 0.71,
    "weak_match": 0.17
  },
  "semantic_similarity": 0.31
}
```

---

# Why This Approach Works

This design separates two important signals:

### Content Quality
Is the content safe and educational for children?

### Intent Alignment
Does the content actually match what the parent asked for?

Combining ML predictions with semantic alignment creates a system that is:

- interpretable
- controllable
- trustworthy for parents

---

# Future Improvements

Planned improvements include:

### Dockerized Deployment
Containerizing the backend so the model runs consistently across environments.

### Faster Inference
Caching the embedding model to avoid repeated loading.

### Larger Dataset
Expanding the training dataset to improve model generalization.

### Feedback-Driven Training
Incorporating real user feedback to improve recommendations.

### Intent Expansion
Automatically expanding parent intentions into richer semantic prompts.

---

# Tech Stack

- Python
- Sentence Transformers
- Scikit-learn
- Logistic Regression
- FastAPI
- Cosine Similarity
- JSON artifacts
- Pickle model storage

---

# Author

Developed as a prototype ML system for **Laroye AI**.

Inspired by probabilistic classification approaches such as **FinBERT** and research work with **Wat.AI**.
