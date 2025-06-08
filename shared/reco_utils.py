def get_unseen_articles(user_id, clicks_df, all_article_ids):
    seen = clicks_df[clicks_df["user_id"] == user_id]["click_article_id"].unique().tolist()
    return [aid for aid in all_article_ids if aid not in seen]

def recommend_articles(user_id, model, clicks_df, metadata_df, top_n=10):
    all_article_ids = metadata_df["article_id"].tolist()
    unseen = get_unseen_articles(user_id, clicks_df, all_article_ids)
    predictions = [model.predict(user_id, aid) for aid in unseen]
    predictions.sort(key=lambda x: x.est, reverse=True)
    return predictions[:top_n]
