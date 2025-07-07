import numpy as np
from typing import List, Dict, Any
from langchain.schema import Document

class KnowledgeBaseEvaluator:
    """Evaluation metrics for knowledge base performance"""
    
    def __init__(self, knowledge_manager):
        self.knowledge_manager = knowledge_manager
    
    def evaluate_chunking_quality(self, documents: List[Document]) -> Dict[str, float]:
        """Evaluate chunking quality metrics"""
        # Consistency check: ensure all documents have 'document_type' in metadata
        for i, doc in enumerate(documents):
            if 'document_type' not in doc.metadata:
                raise ValueError(f"Document at index {i} is missing 'document_type' in metadata: {doc.metadata}")
        chunk_sizes = []
        overlaps = []
        
        for doc in documents:
            chunks, _ = self.knowledge_manager.chunker.chunk_document(doc)
            # Calculate chunk size distribution
            for chunk in chunks:
                chunk_sizes.append(len(chunk.page_content))
            # Calculate overlap ratios
            for i in range(len(chunks) - 1):
                overlap = self._calculate_overlap(chunks[i].page_content, chunks[i+1].page_content)
                overlaps.append(overlap)
        
        return {
            'average_chunk_size': float(np.mean(chunk_sizes)) if chunk_sizes else 0.0,
            'chunk_size_std': float(np.std(chunk_sizes)) if chunk_sizes else 0.0,
            'average_overlap_ratio': float(np.mean(overlaps)) if overlaps else 0.0,
            'total_chunks': len(chunk_sizes)
        }
    
    def _calculate_overlap(self, text1: str, text2: str) -> float:
        """Calculate overlap ratio between two text chunks"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0
    
    def evaluate_retrieval_accuracy(self, test_queries: List[Dict]) -> Dict[str, float]:
        """Evaluate retrieval accuracy using precision, recall, and F1"""
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for query_data in test_queries:
            query = query_data['query']
            expected_types = set(query_data.get('expected_types', []))
            results = self.knowledge_manager.search_knowledge_base(query, k=5)
            retrieved_types = set([doc.metadata.get('document_type') for doc in results])
            # Calculate metrics
            tp = len(expected_types.intersection(retrieved_types))
            fp = len(retrieved_types - expected_types)
            fn = len(expected_types - retrieved_types)
            true_positives += tp
            false_positives += fp
            false_negatives += fn
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def generate_evaluation_report(self, output_path: str = "evaluation_report.html"):
        """Generate comprehensive evaluation report"""
        # Load test data
        test_documents = self.knowledge_manager.load_sample_documents()
        # Evaluate chunking quality
        chunking_metrics = self.evaluate_chunking_quality(test_documents)
        # Test queries for retrieval evaluation
        test_queries = [
            {'query': 'API authentication methods', 'expected_types': ['api_reference', 'tutorial']},
            {'query': 'Privacy policy data collection', 'expected_types': ['policy_document']},
            {'query': 'Application startup troubleshooting', 'expected_types': ['troubleshooting_guide']},
            {'query': 'Support ticket database issues', 'expected_types': ['support_ticket']},
            {'query': 'How to reset password', 'expected_types': ['faq']},
        ]
        # Evaluate retrieval accuracy
        retrieval_metrics = self.evaluate_retrieval_accuracy(test_queries)
        # Generate HTML report
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enterprise Knowledge Management System - Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ background-color: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .header {{ color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }}
                .score {{ font-size: 24px; font-weight: bold; color: #007acc; }}
            </style>
        </head>
        <body>
            <h1 class=\"header\">Enterprise Knowledge Management System - Evaluation Report</h1>
            <h2>Chunking Quality Metrics</h2>
            <div class=\"metric\">
                <h3>Average Chunk Size</h3>
                <p class=\"score\">{chunking_metrics['average_chunk_size']:.0f} characters</p>
                <p>Standard deviation: {chunking_metrics['chunk_size_std']:.0f}</p>
            </div>
            <div class=\"metric\">
                <h3>Average Overlap Ratio</h3>
                <p class=\"score\">{chunking_metrics['average_overlap_ratio']:.2%}</p>
                <p>Total chunks generated: {chunking_metrics['total_chunks']}</p>
            </div>
            <h2>Retrieval Accuracy Metrics</h2>
            <div class=\"metric\">
                <h3>Precision</h3>
                <p class=\"score\">{retrieval_metrics['precision']:.2%}</p>
            </div>
            <div class=\"metric\">
                <h3>Recall</h3>
                <p class=\"score\">{retrieval_metrics['recall']:.2%}</p>
            </div>
            <div class=\"metric\">
                <h3>F1 Score</h3>
                <p class=\"score\">{retrieval_metrics['f1_score']:.2%}</p>
            </div>
            <h2>System Performance Summary</h2>
            <div class=\"metric\">
                <h3>Overall Performance</h3>
                <p>True Positives: {retrieval_metrics['true_positives']}</p>
                <p>False Positives: {retrieval_metrics['false_positives']}</p>
                <p>False Negatives: {retrieval_metrics['false_negatives']}</p>
            </div>
            <h2>Recommendations</h2>
            <ul>
                <li>{'✅ Chunking performance is optimal' if chunking_metrics['average_chunk_size'] < 1000 else '⚠️ Consider reducing chunk sizes'}</li>
                <li>{'✅ Overlap ratio is appropriate' if 0.1 <= chunking_metrics['average_overlap_ratio'] <= 0.3 else '⚠️ Adjust overlap ratios'}</li>
                <li>{'✅ Retrieval accuracy is good' if retrieval_metrics['f1_score'] > 0.7 else '⚠️ Improve retrieval accuracy'}</li>
            </ul>
        </body>
        </html>
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        return {
            'chunking_metrics': chunking_metrics,
            'retrieval_metrics': retrieval_metrics,
            'report_path': output_path
        } 