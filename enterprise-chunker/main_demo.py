import os
from dotenv import load_dotenv
from .evaluation_metrics import KnowledgeBaseEvaluator
from .dataset_loader import DatasetLoader
from .main import EnterpriseKnowledgeManager  # Updated import for local usage

def run_full_demo():
    """Run complete demonstration of the system"""
    print("🚀 Enterprise Knowledge Management System - Full Demo")
    print("=" * 60)
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key in a .env file or environment variable")
        return
    try:
        # Initialize components
        print("\n1. Initializing system components...")
        knowledge_manager = EnterpriseKnowledgeManager(
            openai_api_key=api_key,
            chroma_db_path="./demo_knowledge_db"
        )
        evaluator = KnowledgeBaseEvaluator(knowledge_manager)
        dataset_loader = DatasetLoader()
        # Load datasets
        print("\n2. Loading datasets...")
        sample_docs = knowledge_manager.load_sample_documents()
        enterprise_docs = dataset_loader.load_sample_enterprise_data()
        support_tickets = dataset_loader.create_synthetic_support_tickets(5)
        all_docs = sample_docs + enterprise_docs + support_tickets
        print(f"   Total documents loaded: {len(all_docs)}")
        # Process documents
        print("\n3. Processing documents with adaptive chunking...")
        processing_stats = knowledge_manager.process_documents(all_docs)
        print(f"   ✅ Processed {processing_stats['total_documents']} documents")
        print(f"   ✅ Created {processing_stats['total_chunks']} chunks")
        print(f"   ✅ Document types: {processing_stats['document_types']}")
        # Run evaluation
        print("\n4. Running comprehensive evaluation...")
        evaluation_results = evaluator.generate_evaluation_report("demo_evaluation_report.html")
        print(f"   ✅ Evaluation complete!")
        print(f"   ✅ Report saved to: {evaluation_results['report_path']}")
        # Display key metrics
        print("\n5. Key Performance Metrics:")
        retrieval_metrics = evaluation_results['retrieval_metrics']
        print(f"   📊 Precision: {retrieval_metrics['precision']:.2%}")
        print(f"   📊 Recall: {retrieval_metrics['recall']:.2%}")
        print(f"   📊 F1 Score: {retrieval_metrics['f1_score']:.2%}")
        # Interactive search demo
        print("\n6. Interactive Search Demo:")
        demo_queries = [
            "How to configure database connections?",
            "API rate limiting information",
            "Support ticket about login issues",
            "Application performance problems"
        ]
        for query in demo_queries:
            print(f"\n   🔍 Query: '{query}'")
            results = knowledge_manager.search_knowledge_base(query, k=2)
            for i, result in enumerate(results, 1):
                doc_type = result.metadata.get('document_type', 'unknown')
                source = result.metadata.get('source', 'unknown')
                preview = result.page_content[:150].replace('\n', ' ')
                print(f"      {i}. [{doc_type}] {source}")
                print(f"         {preview}...")
        # Export final results
        print("\n7. Exporting results...")
        knowledge_manager.export_processed_data("demo_final_results.json")
        print("\n🎉 Demo completed successfully!")
        print(f"📄 View evaluation report: demo_evaluation_report.html")
        print(f"📊 View detailed results: demo_final_results.json")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required packages: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    load_dotenv()
    run_full_demo() 