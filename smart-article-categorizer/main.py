import os
from src.data_preparation import DataPreprocessor
from src.chroma_storage import ChromaDataManager
from src.classification import ArticleClassifier
from src.evaluation import ModelEvaluator

def main():
    # Initialize components
    data_prep = DataPreprocessor()
    classifier = ArticleClassifier()
    evaluator = ModelEvaluator()
    chroma_manager = ChromaDataManager()
    
    # Prepare data
    print("Preparing data...")
    X_train, X_test, y_train, y_test = data_prep.prepare_data()
    
    # Initialize embedders
    openai_key = os.getenv('OPENAI_API_KEY')  # Set this in environment
    classifier.initialize_embedders(openai_key)
    
    # Train models
    embedding_types = ['word2vec', 'bert', 'sentence_bert']
    if openai_key:
        embedding_types.append('openai')
    
    for embedding_type in embedding_types:
        print(f"Training {embedding_type} model...")
        classifier.train_classifier(X_train, y_train, embedding_type)
        
        # Evaluate
        predictions, _ = classifier.predict(X_test, embedding_type)
        metrics = evaluator.evaluate_model(y_test, predictions, embedding_type)
        print(f"{embedding_type} metrics: {metrics}")
    
    # Save models
    classifier.save_models('./models')
    
    # Generate comparison report
    comparison_df = evaluator.compare_models()
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Plot comparison
    evaluator.plot_model_comparison()

if __name__ == "__main__":
    main()