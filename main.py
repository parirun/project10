from flask import Flask, request, jsonify
import pandas as pd
import pickle
from pathlib import Path
from surprise import SVD
from reco_utils import recommend_articles, get_unseen_articles  # Ton module utilitaire

app = Flask(__name__)

# Chargement du modèle SVD (déjà entraîné et sauvegardé)
MODEL_PATH = Path(__file__).parent / "shared" / "svd_model.pkl"
METADATA_PATH = Path(__file__).parent / "shared" / "articles_metadata.csv"
CLICKS_DIR = Path(__file__).parent / "shared"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"❌ Erreur de chargement du modèle : {e}")

try:
    metadata = pd.read_csv(METADATA_PATH)
except Exception as e:
    metadata = None
    print(f"❌ Erreur de chargement du fichier metadata : {e}")

# Chargement des clics utilisateur
click_files = list(CLICKS_DIR.glob("clicks_hour_*.csv"))
try:
    clicks_df = pd.concat([pd.read_csv(f) for f in click_files], ignore_index=True)
    clicks_df["rating"] = 1.0
except Exception as e:
    clicks_df = None
    print(f"❌ Erreur chargement clics : {e}")


@app.route("/recommend", methods=["POST"])
def recommend():
    if not model or clicks_df is None or metadata is None:
        return jsonify({"error": "❌ Modèle ou données non chargés"}), 500

    try:
        data = request.get_json()
        user_id = int(data["user_id"])
    except Exception:
        return jsonify({"error": "❌ Format JSON attendu : { 'user_id': 123 }"}), 400

    try:
        reco = recommend_articles(user_id, model, clicks_df, metadata, top_n=5)
        output = [{"article_id": int(pred.iid), "score": round(pred.est, 2)} for pred in reco]
        return jsonify(output)
    except Exception as e:
        return jsonify({"error": f"❌ Erreur recommandation : {str(e)}"}), 500


@app.route("/")
def home():
    return "✅ Moteur de recommandation SVD en ligne !"

if __name__ == "__main__":
    app.run(debug=True)
