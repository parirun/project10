import logging
import azure.functions as func
import json
import pandas as pd
from pathlib import Path
import pickle
from shared.reco_utils import recommend_articles, get_unseen_articles


def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        data = req.get_json()
        user_id = int(data.get("user_id"))
    except Exception:
        return func.HttpResponse("❌ Format JSON attendu : { 'user_id': 123 }", status_code=400)

    # 1. Charger les clics
    clicks_dir = Path(__file__).parent.parent / "shared"
    clicks_files = list(clicks_dir.glob("clicks_hour_*.csv"))
    if not clicks_files:
        return func.HttpResponse("❌ Aucun fichier clics trouvé.", status_code=500)

    clicks_df = pd.concat([pd.read_csv(f) for f in clicks_files], ignore_index=True)
    clicks_df["rating"] = 1.0

    # 2. Charger metadata
    metadata_path = clicks_dir / "articles_metadata.csv"
    if not metadata_path.exists():
        return func.HttpResponse("❌ Fichier metadata manquant.", status_code=500)
    metadata = pd.read_csv(metadata_path)

    # 3. Charger modèle SVD préentraîné
    model_path = clicks_dir / "svd_model.pkl"
    if not model_path.exists():
        return func.HttpResponse("❌ Modèle SVD non trouvé.", status_code=500)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 4. Recommandation
    try:
        reco = recommend_articles(user_id, model, clicks_df, metadata, top_n=5)
        output = [{"article_id": pred.iid, "score": round(pred.est, 2)} for pred in reco]
        return func.HttpResponse(json.dumps(output), mimetype="application/json")
    except Exception as e:
        logging.exception("Erreur lors de la recommandation")
        return func.HttpResponse(f"❌ Erreur interne : {type(e).__name__} - {str(e)}", status_code=500)