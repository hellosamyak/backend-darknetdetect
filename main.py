# Install dependencies first:
# pip install scikit-learn rapidfuzz

from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ------------------------------
# 1. Slang Lexicon
# ------------------------------
slang_dict = {
    "yayo": "cocaine",
    "beans": "ecstasy (MDMA)",
    "xanny": "xanax",
    "green": "marijuana",
    "ice": "meth",
    "cocaine": "cocaine",
    "weed": "marijuana",
    "ganja": "marijuana",
    "smack": "heroin"
}

# ------------------------------
# 2. Slang Normalizer
# ------------------------------
def normalize_text(text, slang_dict, threshold=70):
    text_lower = text.lower()
    words = text_lower.split()  # keep hyphens intact
    normalized_words = []
    
    for word in words:
        replaced = False
        for slang, meaning in slang_dict.items():
            # direct match or fuzzy match
            if slang == word or fuzz.ratio(slang, word) >= threshold:
                normalized_words.append(meaning)  # replace with standard name
                replaced = True
                break
        if not replaced:
            normalized_words.append(word)
    
    return " ".join(normalized_words)

# ------------------------------
# 3. Example Dataset (toy version)
# Label: 1 = drug-related, 0 = clean
# ------------------------------
texts = [
    "I got some yayo for sale",
    "Selling beans cheap tonight",
    "Anyone need c0ca1n3?",
    "He wants to buy green",
    "Got xannyyy if you want",
    "I smoke weed every day",
    "Snow is falling outside",   # NOT drug-related
    "This party has no drugs",   # NOT drug-related
    "I need some food for dinner", # NOT drug-related
    "She loves green tea"        # NOT drug-related
]

labels = [1,1,1,1,1,1,0,0,0,0]

# ------------------------------
# 4. Normalize Dataset
# ------------------------------
normalized_texts = [normalize_text(t, slang_dict) for t in texts]

# ------------------------------
# 5. Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    normalized_texts, labels, test_size=0.3, random_state=42
)

# ------------------------------
# 6. ML Pipeline: TF-IDF + Logistic Regression
# ------------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=200))
])

pipeline.fit(X_train, y_train)

# ------------------------------
# 7. Evaluate
# ------------------------------
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

# ------------------------------
# 8. Try New Examples
# ------------------------------
new_texts = [
    "selling ya-yo at good price",
    "let's drink some green tea",
    "got ice for party",
    "need some smack asap",
    "snow on the mountain"
]

# normalize them
new_texts_norm = [normalize_text(t, slang_dict) for t in new_texts]

predictions = pipeline.predict(new_texts_norm)

for text, pred in zip(new_texts, predictions):
    label = "⚠️ Drug-related" if pred == 1 else "✅ Clean"
    print(f"{label}: {text}")
