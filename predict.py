import sys
import json
import joblib
from rapidfuzz import fuzz
from pathlib import Path

MODEL_PATH = Path("models/drug_slang_model.pkl")
SLANG_JSON = Path("models/slang_dict.json")

def normalize_text(text: str, slang_map: dict, threshold: int = 70) -> str:
    text_lower = str(text).lower()
    words = text_lower.split()
    out = []
    for w in words:
        replaced = False
        for slang, meaning in slang_map.items():
            if slang == w or fuzz.ratio(slang, w) >= threshold:
                out.append(meaning)
                replaced = True
                break
        if not replaced:
            out.append(w)
    return " ".join(out)

def main():
    if not MODEL_PATH.exists() or not SLANG_JSON.exists():
        print("❌ Model or slang dict not found. Train first: python train_model.py")
        return

    pipeline = joblib.load(MODEL_PATH)
    slang_dict = json.loads(SLANG_JSON.read_text(encoding="utf-8"))

    # Get text either from command line or prompt
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter a message: ").strip()

    norm = normalize_text(text, slang_dict)
    pred = pipeline.predict([norm])[0]
    label = "⚠️ Drug-related" if pred == 1 else "✅ Clean"

    print(f"\nText: {text}")
    print(f"Normalized: {norm}")
    print(f"Prediction: {label}")

if __name__ == "__main__":
    main()
