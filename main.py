import sys
import json
import logging
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify
import pickle


# Ajout du dossier shared/ au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent / "shared"))

from reco_utils import recommend_articles, get_unseen_articles

app = Flask(__name__)

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        user_id = int(data.get("user_id"))
    except Exception:
        return jsonify({"error": "❌ Format JSON attendu : { 'user_id': 123 }"}), 400

    # 📂 Chargement des clics
    clicks_dir = Path(__file__).resolve().parent / "shared"
    clicks_files = list(clicks_dir.glob("clicks_hour_*.csv"))
    if not clicks_files:
        return jsonify({"error": "❌ Aucun fichier clics trouvé."}), 500

    clicks_df = pd.concat([pd.read_csv(f) for f in clicks_files], ignore_index=True)
    clicks_df["rating"] = 1.0

    # 📂 Chargement du fichier metadata
    metadata_path = clicks_dir / "articles_metadata.csv"
    if not metadata_path.exists():
        return jsonify({"error": "❌ Fichier metadata manquant."}), 500

    metadata = pd.read_csv(metadata_path)

    # 📂 Chargement du modèle SVD préentraîné
    model_path = clicks_dir / "svd_model.pkl"
    if not model_path.exists():
        return jsonify({"error": "❌ Modèle SVD non trouvé."}), 500

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        logging.exception("Erreur de chargement du modèle")
        return jsonify({"error": f"❌ Erreur de chargement du modèle : {str(e)}"}), 500

    # 🎯 Génération des recommandations
    try:
        reco = recommend_articles(user_id, model, clicks_df, metadata, top_n=5)
        output = [{"article_id": pred.iid, "score": round(pred.est, 2)} for pred in reco]
        return jsonify(output), 200
    except Exception as e:
        logging.exception("Erreur lors de la recommandation")
        return jsonify({"error": f"❌ Erreur interne : {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
