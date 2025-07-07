# Indian Legal Document Search System

A comprehensive system for searching Indian legal documents, comparing four advanced similarity methods, and evaluating their effectiveness for legal document retrieval.

---

## 🚩 Project Deliverables Checklist

- [x] **Working code** (Python, Streamlit)
- [x] **UI:** Web app with 4-method comparison, document upload, and metrics dashboard
- [x] **Analysis:** Interactive performance metrics and support for user-driven evaluation

---

## 1. Similarity Methods Implemented

- **Cosine Similarity:** Standard semantic matching
- **Euclidean Distance:** Geometric distance in embedding space
- **MMR (Maximal Marginal Relevance):** Reduces redundancy in results
- **Hybrid Similarity:** 0.6 × Cosine + 0.4 × Legal_Entity_Match (domain-specific)

All four methods are implemented and compared side-by-side in the UI.

---

## 2. Test Dataset

The system is designed to work with:
- Indian Income Tax Act sections (sample PDF included)
- GST Act provisions (add your own PDF/Word files)
- Sample court judgments (add your own PDF/Word files)
- Property law documents (add your own PDF/Word files)

**How to add documents:**
- Upload via the web UI (PDF or Word)
- Or, place files in `legal-doc-search-system/docs/` and run the batch ingestion script

---

## 3. Comparison Framework

- **Precision@5:** Fraction of top 5 results that are relevant (user-marked)
- **Recall@5:** Fraction of all relevant docs that appear in top 5 results
- **Diversity Score:** Measures variety in results (especially for MMR)
- **Side-by-side UI:** All 4 methods shown in columns for direct comparison

Metrics are calculated interactively based on user-marked relevance for each query.

---

## 4. Web UI Interface

- **Document upload:** Supports PDF and Word (docx) files
- **Text query input:** Enter any legal query
- **4-column results comparison:** Cosine, Euclidean, MMR, Hybrid
- **Performance metrics dashboard:** Precision, Recall, Diversity (user-driven)
- **Interactive relevance marking:** Mark results as relevant to update metrics

---

## 5. Test Queries

Try these queries to evaluate the system:
- "Income tax deduction for education"
- "GST rate for textile products"
- "Property registration process"
- "Court fee structure"

---

## 6. How to Use

### Setup

1. **Install dependencies:**
   ```bash
   pip install streamlit langchain_community scikit-learn
   ```
2. **(Optional) Add documents:**
   - Upload via UI, or
   - Place files in `docs/` and run:
     ```bash
     python legal-doc-search-system/ingest_and_index.py
     ```
3. **Run the app:**
   ```bash
   streamlit run legal-doc-search-system/app/app.py
   ```

### Usage

1. **Upload legal documents** (PDF/Word)
2. **Ingest documents** (click "Ingest Documents")
3. **Enter a legal query** and click "Search"
4. **Mark relevant results** using checkboxes
5. **Click "Calculate Metrics"** to view Precision@5, Recall@5, and Diversity for each method
6. **Compare methods** side-by-side and use metrics for your analysis/report

---

## 7. Analysis & Recommendations

- Use the metrics dashboard to compare retrieval quality for each method on your queries.
- Mark relevant results to get real precision/recall.
- For a performance report, run several queries, mark relevance, and record the metrics for each method.
- Use the side-by-side UI to visually assess diversity and redundancy in results.

---

## 8. Requirements
- Python 3.8+
- `streamlit`, `langchain_community`, `scikit-learn`

---

## 9. License
MIT License 