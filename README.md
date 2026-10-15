---
title: career_chatbot
app_file: app.py
sdk: gradio
sdk_version: 5.34.2
---
## Semantic Retrieval Baseline

### Objective

Establish a reproducible baseline for semantic retrieval before comparing it with hybrid retrieval using BM25 and Reciprocal Rank Fusion (RRF).

The baseline uses:
- **Vector store:** FAISS
- **Embedding model:** OpenAI `text-embedding-3-small`
- **Evaluation dataset:** A curated set of retrieval questions with relevant chunk IDs
- **Metrics:** Hit Rate@k, Precision@k, Recall@k, and Mean Reciprocal Rank (MRR)
- **Outputs:** Per-question results and aggregate metrics saved as CSV files

### Setup

Activate the project's virtual environment and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Configure your OpenAI API key in a local `.env` file. Never commit `.env` or expose API keys in logs or documentation.

### Reproducing the baseline

The evaluation script loads an existing local FAISS index and evaluates retrieval against a relevance-labeled CSV dataset.

**Important:** The original index, source documents, chunk metadata, and evaluation cases are specific to a private career-profile corpus and are not included in the public repository.

#### Local reproduction

1. Configure your OpenAI API key in a local `.env` file.
2. Prepare an approved corpus and a matching evaluation dataset with relevant chunk IDs.
3. Build the local FAISS index from the prepared corpus:

   ```bash
   python app/vector_store.py --rebuild
   ```

4. Run the semantic retrieval evaluation:

   ```bash
   python app/evaluation.py --k 4
   python app/evaluation.py --k 8
   ```

5. Review the generated per-question results and aggregate metrics locally.

The indexing command must be run after preparing the corpus and configuring the project's expected input sources. The evaluator expects the saved index and evaluation CSV to exist at the paths configured by the project.

#### Privacy

Do not publish personal source documents, private chunk text or metadata, saved FAISS indexes, pickle files, API keys, or evaluation outputs containing personal information. For a public reproduction, use synthetic or explicitly approved data and matching relevance labels.

The reported metrics below are from the original eight-question experiment on the private corpus. They are reference results, not a guarantee that a different corpus will produce the same values.

### Evaluation metrics

- **Hit Rate@k:** Fraction of questions for which at least one relevant chunk appears in the top k results.
- **Precision@k:** Fraction of the top k retrieved chunks that are relevant.
- **Recall@k:** Fraction of all labeled relevant chunks retrieved in the top k results.
- **MRR:** Mean reciprocal rank of the first relevant result for each question.

### Baseline reference

The initial semantic retrieval experiment used eight manually curated questions.

| Metric | k = 4 | k = 8 |
|---|---:|---:|
| Hit Rate@k | 0.875 | 1.000 |
| Mean Precision@k | 0.406 | 0.250 |
| Mean Recall@k | Not recorded in this reference | 0.844 |
| MRR | 0.812 | 0.838 |

These figures are an initial reference for this evaluation dataset, not a general claim about retrieval quality. Results should be interpreted alongside the individual queries and their relevance labels.

### Tests

Run the unit tests with:

```bash
python -m pytest -q
```

### Privacy and reproducibility

The repository should contain the retrieval implementation, evaluation logic, tests, and only reviewed, sanitized benchmark data. Private profile documents, generated chunks containing personal information, saved indexes containing private text or metadata, environment files, and credentials must remain local.

To reproduce the experiment, prepare an equivalent approved corpus and relevance-labeled evaluation dataset, build the index locally, and run the evaluator using the commands above.