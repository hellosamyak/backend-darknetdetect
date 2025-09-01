import json
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from pathlib import Path

# ------------------------------
# Paths
# ------------------------------
DATA_CSV = Path("data/drug_slang_dataset.csv")
MODEL_PATH = Path("models/drug_slang_model.pkl")
SLANG_JSON = Path("models/slang_dict.json")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------
# Slang dictionary (expandable)
# ------------------------------
slang_dict = {
    # Cocaine / Crack
    "yayo": "cocaine", "yeyo": "cocaine", "coke": "cocaine", "snow": "cocaine", "blow": "cocaine",
    "flake": "cocaine", "charlie": "cocaine", "white girl": "cocaine", "powder": "cocaine",
    "rock": "crack cocaine", "hard": "crack cocaine", "ready rock": "crack cocaine",

    # Heroin
    "smack": "heroin", "dope": "heroin", "brown sugar": "heroin", "china white": "heroin",
    "tar": "heroin", "horse": "heroin", "junk": "heroin", "h": "heroin",

    # Marijuana
    "weed": "marijuana", "ganja": "marijuana", "kush": "marijuana", "420": "marijuana",
    "pot": "marijuana", "reefer": "marijuana", "bud": "marijuana", "grass": "marijuana",
    "mary jane": "marijuana", "skunk": "marijuana", "herb": "marijuana",

    # Meth
    "ice": "meth", "crystal": "meth", "glass": "meth", "tina": "meth",
    "crank": "meth", "speed": "meth", "chalk": "meth",

    # MDMA
    "molly": "ecstasy", "ecstasy": "ecstasy", "e": "ecstasy", "x": "ecstasy",
    "beans": "ecstasy", "rolls": "ecstasy", "adam": "ecstasy", "xtc": "ecstasy",

    # LSD & Mushrooms
    "acid": "lsd", "tabs": "lsd", "blotter": "lsd", "lucy": "lsd", "dots": "lsd",
    "shrooms": "mushrooms", "boomers": "mushrooms", "magic": "mushrooms",

    # Pills / Opioids / Sedatives
    "oxy": "oxycodone", "percs": "percocet", "perc": "percocet", "perkies": "percocet",
    "hydro": "hydrocodone", "vikes": "vicodin", "vicodin": "vicodin",
    "xanax": "xanax", "xanny": "xanax", "zanny": "xanax", "bars": "xanax",

    # Fentanyl & Lean
    "fent": "fentanyl", "fenty": "fentanyl", "china girl": "fentanyl",
    "lean": "codeine", "purple drank": "codeine", "sizzurp": "codeine", "syrup": "codeine"
}

def normalize_text(text: str, slang_map: dict, threshold: int = 70) -> str:
    """
    Replace slang tokens with their canonical drug names using fuzzy matching.
    """
    text_lower = str(text).lower()
    words = text_lower.split()  # keep hyphens; we're matching tokens
    out = []
    for w in words:
        replaced = False
        for slang, meaning in slang_map.items():
            # exact token match or fuzzy match
            if slang == w or fuzz.ratio(slang, w) >= threshold:
                out.append(meaning)
                replaced = True
                break
        if not replaced:
            out.append(w)
    return " ".join(out)

def main():
    # 1) Load data
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Could not find {DATA_CSV}. Create it first.")

    df = pd.read_csv(DATA_CSV)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have columns: text,label")

    # 2) Normalize text
    df["normalized"] = df["text"].apply(lambda t: normalize_text(t, slang_dict))

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["normalized"], df["label"], test_size=0.3, random_state=42
    )

    # 4) Build & train model (TF-IDF + Logistic Regression)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=500))
    ])
    pipeline.fit(X_train, y_train)

    # 5) Evaluate
    y_pred = pipeline.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 6) Save model + slang dictionary
    joblib.dump(pipeline, MODEL_PATH)
    with open(SLANG_JSON, "w", encoding="utf-8") as f:
        json.dump(slang_dict, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved model to: {MODEL_PATH}")
    print(f"✅ Saved slang dict to: {SLANG_JSON}")

if __name__ == "__main__":
    main()
