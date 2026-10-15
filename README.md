---

## title: career_chatbot
app_file: app.py
sdk: gradio
sdk_version: 5.34.2

## Retrieval Architecture

The project supports two retrieval modes:

- **Semantic retrieval:** Uses OpenAI `text-embedding-3-small` embeddings and a persisted FAISS index to retrieve semantically similar chunks.
- **Hybrid retrieval:** Combines semantic rankings with BM25 keyword rankings using Reciprocal Rank Fusion (RRF).

RRF combines ranked lists using reciprocal rank scores. It allows semantic similarity and lexical matching to contribute without directly comparing their raw scores.

## Setup

Activate the project's virtual environment and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Configure your OpenAI API key in a local `.env` file. Never commit `.env` or expose API keys in logs or documentation.

## Evaluation

The evaluation script loads an existing local FAISS index and evaluates retrieval against a relevance-labeled CSV dataset. It reports Hit Rate@k, Precision@k, Recall@k, and Mean Reciprocal Rank (MRR), with per-question results and aggregate metrics saved as CSV files.

The original index, source documents, chunk metadata, and evaluation cases are specific to a private career-profile corpus and are not included in the public repository.

### Reproducing the semantic baseline

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



### Running hybrid retrieval evaluation

The hybrid evaluator uses the same semantic index and evaluation dataset, adding BM25 keyword rankings and combining them with semantic rankings using RRF.

```bash
python app/evaluation.py --mode hybrid --k 4
python app/evaluation.py --mode hybrid --k 8
```

Semantic and hybrid evaluation outputs are written to separate files so that one mode does not overwrite the other. Review the generated CSVs locally; do not commit evaluation outputs containing personal information.

The indexing command must be run after preparing the corpus and configuring the project's expected input sources. The evaluator expects the saved index and evaluation CSV to exist at the paths configured by the project.

## Evaluation Metrics

- **Hit Rate@k:** Fraction of questions for which at least one relevant chunk appears in the top k results.
- **Precision@k:** Fraction of the top k retrieved chunks that are relevant.
- **Recall@k:** Fraction of all labeled relevant chunks retrieved in the top k results.
- **MRR:** Mean reciprocal rank of the first relevant result for each question.



## Results

The results below are from eight manually curated questions on the private career-profile corpus. They are reference results for this dataset, not a guarantee that a different corpus will produce the same values.

### Semantic retrieval baseline


| Metric           | k = 4                          | k = 8 |
| ---------------- | ------------------------------ | ----- |
| Hit Rate@k       | 0.875                          | 1.000 |
| Mean Precision@k | 0.406                          | 0.250 |
| Mean Recall@k    | Not recorded in this reference | 0.844 |
| MRR              | 0.812                          | 0.838 |




### Hybrid retrieval: BM25 + RRF


| Metric           | k = 4 | k = 8 |
| ---------------- | ----- | ----- |
| Hit Rate@k       | 0.875 | 1.000 |
| Mean Precision@k | 0.312 | 0.281 |
| Mean Recall@k    | 0.646 | 0.938 |
| MRR              | 0.875 | 0.900 |




### Interpretation and current limitations

Hybrid retrieval improves MRR at both cutoffs and improves precision and recall at k = 8 in this experiment. At k = 4, precision decreases while hit rate remains unchanged. These results suggest that hybrid retrieval is promising, but gains are not consistent across all metrics and questions.

A known limitation remains: broad questions can rank generic profile-summary chunks above specific project evidence. Combining semantic and keyword rankings does not fully resolve this issue when both retrieval methods rank relevant chunks poorly. The next experiment is query-aware retrieval, which will use question intent to prioritize appropriate evidence and will be evaluated against the same benchmark. A reranker may be considered if ranking errors remain.

These figures should be interpreted alongside the individual queries and their relevance labels. Aggregate metrics alone do not guarantee that the generated answer will be complete or grounded in the right evidence.

## Tests

Run the unit tests with:

```bash
python -m pytest -q
```



## Privacy and Reproducibility

Do not publish personal source documents, private chunk text or metadata, saved FAISS indexes, pickle files, API keys, or evaluation outputs containing personal information. The repository should contain the retrieval implementation, evaluation logic, tests, and only reviewed, sanitized benchmark data. Private profile documents, generated chunks containing personal information, saved indexes containing private text or metadata, environment files, credentials, and local evaluation results must remain local.

For a public reproduction, use synthetic or explicitly approved data and matching relevance labels. Prepare an equivalent approved corpus and relevance-labeled evaluation dataset, build the index locally, and run the evaluator using the commands above.