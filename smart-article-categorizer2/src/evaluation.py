import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        
    def evaluate_model(self, y_true, y_pred, model_name):
        """Evaluate model performance"""
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        return self.results[model_name]
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, categories):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=categories, yticklabels=categories)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def compare_models(self):
        """Compare all models"""
        df = pd.DataFrame(self.results).T
        return df
    
    def plot_model_comparison(self):
        """Plot model comparison"""
        df = self.compare_models()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            df[metric].plot(kind='bar', ax=ax)
            ax.set_title(f'{metric.title()} Comparison')
            ax.set_xlabel('Models')
            ax.set_ylabel(metric.title())
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()