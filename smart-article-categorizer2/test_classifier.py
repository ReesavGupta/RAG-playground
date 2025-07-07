import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.classification import ArticleClassifier

def test_classifier():
    """Test the classifier with sample articles"""
    
    # Initialize classifier
    classifier = ArticleClassifier()
    
    # Load models manually (same as StreamlitApp)
    print("Loading models...")
    classifier.embedders = {}
    classifier.initialize_embedders()
    
    # Load models
    for model_type in ['word2vec', 'bert', 'sentence_bert']:
        model_path = f'./models/{model_type}_classifier.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                classifier.models[model_type] = pickle.load(f)
    
    # Load label encoder
    label_path = './models/label_encoder.pkl'
    if os.path.exists(label_path):
        with open(label_path, 'rb') as f:
            classifier.label_encoder = pickle.load(f)
    
    # Test articles
    test_articles = [
        "Apple announces new iPhone with advanced AI features and improved camera system",
        "Goldman Sachs reports record quarterly profits despite market uncertainty",
        "New cancer treatment shows promising results in clinical trials",
        "Basketball championship finals draw record viewers worldwide",
        "Senate passes new infrastructure bill with bipartisan support",
        "Hollywood blockbuster breaks box office records worldwide"
    ]
    
    expected_categories = ['Tech', 'Finance', 'Healthcare', 'Sports', 'Politics', 'Entertainment']
    
    print("\nTesting classifier predictions:")
    print("=" * 50)
    
    for i, (article, expected) in enumerate(zip(test_articles, expected_categories)):
        print(f"\nTest {i+1}:")
        print(f"Article: {article[:50]}...")
        print(f"Expected: {expected}")
        
        # Test each model
        for model_type in ['word2vec', 'bert', 'sentence_bert']:
            try:
                predictions, probabilities = classifier.predict([article], model_type)
                confidence = max(probabilities[0])
                print(f"{model_type.upper()}: {predictions[0]} (confidence: {confidence:.2%})")
            except Exception as e:
                print(f"{model_type.upper()}: Error - {e}")
        
        print("-" * 30)

if __name__ == "__main__":
    test_classifier() 