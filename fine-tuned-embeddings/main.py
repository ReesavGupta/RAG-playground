from src.data_preparation import load_and_prepare_data
from src.fine_tune_embeddings import fine_tune
from src.vector_store import build_vectorstore
from src.predict_conversion import predict_conversion
import pandas as pd

if __name__ == "__main__":
    df = load_and_prepare_data("data/sales_transcripts_110.csv")

    # Fine-tune and build vector store with fine-tuned model
    fine_tune(df)
    build_vectorstore(df, "models/fine_tuned_sales_embed")

    # Build vector store with generic (non-fine-tuned) model
    build_vectorstore(df, "sentence-transformers/all-MiniLM-L6-v2", persist_dir="chroma_db_baseline")

    # Prepare test queries and true labels
    test_queries = df.explode('chunks')['chunks'].tolist()
    test_labels = df.explode('chunks')['label'].tolist()

    # Predict with fine-tuned model
    fine_tuned_preds = [predict_conversion(q, "models/fine_tuned_sales_embed") for q in test_queries]
    # Predict with generic model
    baseline_preds = [predict_conversion(q, "sentence-transformers/all-MiniLM-L6-v2", persist_dir="chroma_db_baseline") for q in test_queries]

    # Binarize predictions with threshold 0.5
    fine_tuned_bin = [1 if p >= 0.5 else 0 for p in fine_tuned_preds]
    baseline_bin = [1 if p >= 0.5 else 0 for p in baseline_preds]

    # Save results for evaluation
    pd.DataFrame({
        'true_label': test_labels,
        'fine_tuned_pred': fine_tuned_bin,
        'baseline_pred': baseline_bin
    }).to_csv('evaluate/model_comparison.csv', index=False)

    print("Saved predictions for both models to evaluate/model_comparison.csv. Use the evaluation script to compare performance.")
