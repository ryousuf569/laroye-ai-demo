import json
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# which sentence-transformer model to use
EMBED_MODEL = "all-MiniLM-L6-v2"

# load the dataset
df = pd.read_csv(r".\data\training_data.csv")
print(f"Loaded {len(df)} rows  •  Labels: {df['label'].unique().tolist()}")

# combine all text columns into one string per row
df["combined_text"] = ("age: " + df["age"].astype(str) + " | "
                       "intentions: " + df["intentions"].fillna("") + " | "
                       "title: " + df["title"].fillna("") + " | "
                       "description: " + df["description"].fillna(""))

X = df["combined_text"]
y = df["label"]

# split 80/20 train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# encode text with SentenceTransformer
print(f"Encoding with {EMBED_MODEL}...")
embedder = SentenceTransformer(EMBED_MODEL)
X_train_vec = embedder.encode(X_train.tolist(), show_progress_bar=True)
X_test_vec = embedder.encode(X_test.tolist(), show_progress_bar=True)

# LogisticRegression classifier
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# accuracy score
y_pred = model.predict(X_test_vec)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2%}\n")
print(classification_report(y_test, y_pred))

# save the model and metadata
with open(r".\artifacts\model.pkl", "wb") as f:
    pickle.dump(model, f)

with open(r".\artifacts\metadata.json", "w") as f:
    json.dump({"embed_model": EMBED_MODEL}, f, indent=2)

print("Saved model.pkl and metadata.json ✓")
