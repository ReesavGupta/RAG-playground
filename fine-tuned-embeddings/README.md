b# Fine-Tuned Embeddings for Sales Conversion Prediction

## Overview
Sales teams often struggle to accurately predict customer conversion likelihood from call transcripts, relying on subjective human judgment. Generic embeddings fail to capture domain-specific sales nuances like buying signals, objection patterns, and conversation dynamics critical for accurate conversion assessment.

This project builds an AI system that fine-tunes embeddings specifically for sales conversations to improve conversion prediction accuracy and enable better customer prioritization.

## Solution Approach
- **Domain-Specific Fine-Tuning:** Fine-tune pre-trained embeddings on sales conversation data with conversion labels to capture sales-specific semantic patterns.
- **Contrastive Learning:** Train embeddings to distinguish between high-conversion and low-conversion conversation patterns.
- **LangChain Framework:** Orchestrate the fine-tuning pipeline, embedding generation, and similarity-based prediction workflow.
- **Evaluation Pipeline:** Compare fine-tuned vs. generic embeddings on conversion prediction tasks.

## Key Inputs
- Call transcripts with conversion outcomes (successful/failed sales)
- Historical sales interaction data
- Customer context and interaction metadata
- Pre-trained embedding models for fine-tuning

## Expected Outputs
- Fine-tuned embedding model optimized for sales conversations
- Conversion probability scores with improved accuracy
- Performance comparison metrics (fine-tuned vs. generic embeddings)
- Deployment-ready prediction system

## Project Structure
```
fine-tuned-embeddings/
├── data/
│   └── sales_transcripts_110.csv         # Labeled sales transcripts
├── models/
│   └── fine_tuned_sales_embed/           # Fine-tuned embedding model
├── chroma_db/                            # Vector store for fine-tuned model
├── chroma_db_baseline/                   # Vector store for baseline model
├── src/
│   ├── data_preparation.py               # Data loading and chunking
│   ├── fine_tune_embeddings.py           # Fine-tuning logic
│   ├── vector_store.py                   # Vector store builder
│   └── predict_conversion.py             # Conversion prediction logic
├── evaluate/
│   ├── evaluate_model.py                 # Evaluation script
│   └── model_comparison.csv              # Predictions for both models
├── main.py                               # Pipeline entry point
└── README.md                             # This file
```

## Setup Instructions
1. **Clone the repository and navigate to the project folder.**
2. **Install dependencies:**
   ```bash
   pip install pandas sentence-transformers langchain torch scikit-learn langchain-huggingface langchain-chroma
   ```
3. **(Optional) Activate your virtual environment.**

## Usage
### 1. Run the Main Pipeline
This will fine-tune the model, build vector stores, generate predictions for both models, and save results for evaluation:
```bash
python main.py
```

### 2. Evaluate Model Performance
This will print accuracy, AUC, and confusion matrix for both the fine-tuned and baseline models:
```bash
python evaluate/evaluate_model.py
```

## Results
**On the provided dataset, both the fine-tuned and baseline models achieved perfect performance:**
```
--- Fine-tuned Model ---
Accuracy: 1.0000
AUC: 1.0000
Confusion Matrix:
[[50  0]
 [ 0 60]]

--- Baseline Model ---
Accuracy: 1.0000
AUC: 1.0000
Confusion Matrix:
[[50  0]
 [ 0 60]]
```
*Note: This is likely due to the simplicity and/or repetition in the sample dataset. For more realistic results, use a larger and more diverse dataset.*

## Deliverables
- Complete fine-tuning implementation with evaluation metrics
- Comparison of fine-tuned vs. generic embeddings
- Deployment-ready, script-based prediction system

## Customization
- To use your own data, replace `data/sales_transcripts_110.csv` with your transcripts (ensure columns: `transcript`, `label`).
- Adjust chunking, model, or evaluation logic as needed in the `src/` folder.

## Contact
For questions or improvements, please open an issue or contact the project maintainer. 