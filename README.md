# vertex-ai-doc-qa

A document Q&A assistant built on Google Cloud Vertex AI and Gemini. Upload PDF documents to Cloud Storage and ask questions about them through a Gradio chat interface. Gemini reads the documents natively — no chunking, no vector database, no embeddings pipeline.

Built as a learning project to explore Vertex AI, but structured as a real application with a clean codebase, CI, and a two-track architecture that can grow into classic AutoML workflows.

![Python](https://img.shields.io/badge/python-3.12-blue)
![Vertex AI](https://img.shields.io/badge/Google_Cloud-Vertex_AI-4285F4)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-blueviolet)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Demo

Ask questions about any PDF — answers include page and section citations sourced directly from the document.

```
Q: What problem does the Attention paper solve?

A: The paper addresses the problem that dominant sequence transduction models,
   based on complex recurrent or convolutional neural networks, are inherently
   sequential, which precludes parallelization within training examples...
   (Abstract, page 1; Introduction, page 2)

Tokens — input: 3,955  output: 105  estimated cost: $0.00066
```

---

## Architecture

Two-phase design:

**Ingest (run once)**
```
Local PDFs → ingest.py → Cloud Storage (gs://bucket/track_genai/data/)
                       → documents.json  (local index)
```

**Query (every question)**
```
Gradio UI → gemini_client.py → Gemini 2.5 Flash (Vertex AI)
                              ↑
                    document URIs from documents.json
                    (Gemini reads PDFs directly from GCS)
```

Gemini's 1M token context window means the entire document set is passed on every request — no retrieval step, no approximate nearest-neighbour search, no information loss from chunking.

---

## Project structure

```
vertex-ai-doc-qa/
├── .env                        # GCP config (not committed)
├── .env.example                # Template for .env
├── requirements.txt
├── README.md
│
├── infra/
│   ├── setup_project.py        # Enable APIs, create GCS bucket (run once)
│   └── auth.py                 # Shared auth helpers
│
├── track_genai/                # Generative AI track (this project)
│   ├── data/                   # PDFs to ingest (not committed)
│   ├── prompts/                # System prompts
│   ├── documents.json          # Auto-generated ingest index (not committed)
│   └── src/
│       ├── ingest.py           # Upload PDFs to Cloud Storage
│       ├── gemini_client.py    # Gemini API wrapper
│       └── app.py              # Gradio UI
│
└── track_automl/               # Classic ML track (planned)
    ├── data/
    └── src/
        ├── train.py
        ├── predict.py
        ├── evaluate.py
        └── pipeline.py
```

---

## Prerequisites

- Python 3.12+
- A Google Cloud account with billing enabled
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/vertex-ai-doc-qa.git
cd vertex-ai-doc-qa
```

### 2. Install the Google Cloud CLI and authenticate

```bash
# macOS
brew install --cask google-cloud-sdk

# All platforms: initialise and authenticate
gcloud init
gcloud auth application-default login \
  --scopes="https://www.googleapis.com/auth/cloud-platform"
```

> **Note:** `gcloud init` and `gcloud auth application-default login` are two separate
> commands that do different things. The first lets *you* use the CLI; the second lets
> your *Python code* authenticate. Both are required.

### 3. Create your environment file

```bash
cp .env.example .env
```

Edit `.env` with your values:

```
GCP_PROJECT_ID=your-project-id       # visible in the GCP console top bar
GCP_REGION=us-central1               # required for gemini-2.5-flash
GCS_BUCKET=your-unique-bucket-name   # created in step 5
```

### 4. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 5. Bootstrap your GCP project (run once)

Enables the Vertex AI, Storage, and IAM APIs, then creates your Cloud Storage bucket:

```bash
python infra/setup_project.py
```

### 6. Grant Vertex AI access to your bucket

> **This step is required.** Without it, Gemini cannot read your PDFs from Cloud Storage
> and every query will return a 500 error.

Vertex AI runs under its own service account and needs explicit read permission on your
bucket. Run the following, replacing `YOUR_PROJECT_ID` and `YOUR_BUCKET`:

```bash
export PROJECT_NUMBER=$(gcloud projects describe YOUR_PROJECT_ID \
  --format="value(projectNumber)")

gcloud storage buckets add-iam-policy-binding gs://YOUR_BUCKET \
  --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-aiplatform.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"
```

You only need to run this once per bucket. To verify it worked:

```bash
gcloud storage buckets get-iam-policy gs://YOUR_BUCKET
```

You should see `service-XXXXXXXXX@gcp-sa-aiplatform.iam.gserviceaccount.com` listed with
`roles/storage.objectViewer`.

### 7. Ingest your documents

Place PDF files in `track_genai/data/` then run:

```bash
python track_genai/src/ingest.py
```

Verify what was ingested:

```bash
python track_genai/src/ingest.py --list
```

### 8. Launch the app

```bash
python track_genai/src/app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

---

## Usage

| Command | Description |
|---|---|
| `python infra/setup_project.py` | Bootstrap GCP project (run once) |
| `python track_genai/src/ingest.py` | Ingest all PDFs in `track_genai/data/` |
| `python track_genai/src/ingest.py --file path/to/doc.pdf` | Ingest a specific file |
| `python track_genai/src/ingest.py --list` | List ingested documents |
| `python track_genai/src/ingest.py --overwrite` | Re-upload existing files |
| `python track_genai/src/app.py` | Launch the Gradio UI |
| `python track_genai/src/gemini_client.py` | Run a standalone API test |

---

## Cost

Using Gemini 2.5 Flash with a typical research paper (~4,000 input tokens per question):

| | Cost |
|---|---|
| Per question | ~$0.0007 |
| 100 questions | ~$0.07 |
| 1,000 questions | ~$0.70 |

Cloud Storage costs for a few PDFs are negligible (well within the free tier).
Set a budget alert in GCP Billing to avoid surprises.

---

## What is implemented

- [x] GCP project bootstrap script (APIs + Cloud Storage)
- [x] PDF ingestion pipeline with local index (`documents.json`)
- [x] Gemini 2.5 Flash integration via Vertex AI
- [x] Multi-turn conversation with session history
- [x] Page and section citations in answers
- [x] Token usage and cost display per query
- [x] Gradio chat UI
- [x] GitHub Actions CI (lint + type check)

## What remains to be done

- [ ] **Track AutoML** — train and deploy a tabular classification model using Vertex AI AutoML
- [ ] **Vertex AI Pipelines** — orchestrate ingest → train → deploy as a reproducible pipeline
- [ ] **Persistent conversation history** — save/load sessions to Firestore or GCS
- [ ] **Multi-document filtering** — let the user select which documents to query per question
- [ ] **Streaming responses** — stream Gemini output token-by-token for a better UX
- [ ] **Deploy to Cloud Run** — containerise the Gradio app and deploy publicly
- [ ] **Evaluation harness** — automated Q&A accuracy tests against known answers

---

## Contributing

This is a learning project but PRs are welcome. Please follow the
[Conventional Commits](https://www.conventionalcommits.org/) spec and ensure
`ruff check` and `mypy` pass before opening a pull request.

---

## License

MIT
