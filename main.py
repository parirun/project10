from flask import Flask, request, jsonify
import pandas as pd
import pickle
from pathlib import Path
from reco_utils import recommend_articles

app = Flask(__name__)

# Chargement des données et du modèle (une seule fois au démarrage)
base_dir = Path(__file__).parent
clicks_dir = base_dir / "model"
clicks_files = list(clicks_dir.glob("clicks_hour_*.csv"))
clicks_df = pd.concat([pd.read_csv(f) for f in clicks_files], ignore_index=True)
clicks_df["rating"] = 1.0

metadata = pd.read_csv(clicks_dir / "articles_metadata.csv")

with open(clicks_dir / "svd_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        user_id = int(data.get("user_id"))
        reco = recommend_articles(user_id, model, clicks_df, metadata, top_n=5)
        output = [{"article_id": pred.iid, "score": round(pred.est, 2)} for pred in reco]
        return jsonify(output)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
