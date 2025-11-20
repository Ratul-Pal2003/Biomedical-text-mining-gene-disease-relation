# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BTM (Biomedical Text Mining) is a Flask web application that extracts gene-disease relationships from biomedical literature using NLP/ML models. The application integrates with PubMed's API and uses specialized scispaCy models for Named Entity Recognition (NER) and BioBERT for relation extraction.

## Development Commands

### Environment Setup
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install scispaCy models (required for NER)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bionlp13cg_md-0.5.3.tar.gz
```

### Running the Application
```bash
# Start Flask development server
python app.py

# Access web interface at http://localhost:5000
```

### Testing
```bash
# Quick test NER models
python test_ner.py

# Quick test BioBERT
python test_biobert.py

# Test health check endpoint
curl http://localhost:5000/health

# Test NER endpoint
curl http://localhost:5000/test_ner

# Test PubMed API integration
curl http://localhost:5000/test_pubmed

# Debug specific abstract processing
curl http://localhost:5000/debug_abstract/<pmid>
```

## Architecture

### Three-Stage Pipeline

1. **Text Acquisition** (app.py, utils/pubmed_api.py)
   - Input: Direct text, PubMed IDs, file uploads (txt/pdf/doc/docx), or entity-based search
   - PubMed integration uses NCBI E-utilities with rate limiting (8.3 req/sec with API key, 2.5 req/sec without)
   - File uploads saved to `data/uploads/` with timestamp prefixes

2. **Entity Recognition** (models/ner_processor.py)
   - Two specialized scispaCy models run in parallel:
     - `en_ner_bc5cdr_md`: Detects diseases and chemicals (BC5CDR dataset)
     - `en_ner_bionlp13cg_md`: Detects genes and proteins (BioNLP13CG dataset)
   - Extracts entities with confidence scores, sentence context, and character positions
   - Returns structured lists: genes, diseases, chemicals

3. **Relation Extraction** (models/biobert_relation_extractor.py, models/semantic_relation_extractor.py)
   - Three approaches available:
     - **Simple NER-based**: Sentence-level co-occurrence (fast, lower precision)
     - **BioBERT-enhanced**: Uses dmis-lab/biobert-v1.1 pre-trained on PubMed/PMC
     - **Semantic embeddings**: Contextual similarity using BioBERT embeddings (computationally expensive)

### BioBERT Relation Extraction Details

The BioBERT approach (primary method) performs:
- Sentence-level splitting and processing
- Gene-disease pair identification within sentences
- Relation type classification using keyword patterns
- Multi-factor confidence scoring (0.0-0.98 range)

**Relation Types Detected**:
- `causative`: Genes that cause diseases
- `risk_factor`: Genes that increase disease risk
- `associated`: General gene-disease associations
- `protective`: Genes that reduce disease risk
- `therapeutic`: Treatment targets

**Confidence Scoring Factors** (biobert_relation_extractor.py:118-158):
- Base score: Minimum of gene and disease confidence
- Boosts: Strong keywords (+0.15), short sentences (+0.1), evidence markers (+0.1), entity proximity (+0.05)
- Capped at 0.98 maximum

### PubMed API Integration

**Key Endpoints** (utils/pubmed_api.py):
- `search_articles()`: Search PubMed using query terms (esearch endpoint)
- `fetch_article_details()`: Retrieve full metadata including title, abstract, authors, journal, year, DOI (efetch endpoint)
- XML parsing handles structured abstracts (Background, Methods, Results, Conclusions)

**Rate Limiting** (pubmed_api.py:34-42):
- With API key: 120ms delay (8.3 req/sec)
- Without API key: 400ms delay (2.5 req/sec)
- Enforced via `_make_request()` with sleep-based throttling

## Configuration

### Environment Variables (.env)
```env
SECRET_KEY=your-secret-key-here
PUBMED_EMAIL=your-email@example.com
PUBMED_API_KEY=your-ncbi-api-key  # Optional but recommended for higher rate limits
```

### Application Settings (config.py)
- File uploads: 16MB max, allowed extensions: txt, pdf, doc, docx
- ML settings: 0.7 default confidence threshold, 10000 max text length
- Processing: Batch size 32, max 1000 relations per request

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Main web interface |
| `/process_text` | POST | Extract relations from text/PMID |
| `/fetch_pubmed` | POST | Fetch abstract by PMID |
| `/search_pubmed` | POST | Search PubMed articles |
| `/search_by_entity` | POST | Search by gene/disease name |
| `/upload` | POST | Upload and process files |
| `/export/<format>` | GET | Export results (JSON/CSV/Excel) |
| `/test_ner` | GET | Test NER functionality |
| `/test_pubmed` | GET | Test PubMed API |
| `/debug_abstract/<pmid>` | GET | Debug abstract processing |
| `/health` | GET | Health check endpoint |

### Key Endpoint Details

**`/process_text`** (app.py:81-185):
- Accepts `text`, `pubmed_id`, `confidence_threshold` (default: 0.7)
- Fetches from PubMed if `pubmed_id` provided
- Runs NER and BioBERT relation extraction
- Saves results to `data/results/` with timestamp
- Returns: relations (filtered by confidence), entities, PubMed metadata

**`/search_by_entity`** (app.py:303-407):
- Accepts `entity`, `entity_type` (gene/disease), `max_papers`, `confidence_threshold`
- Searches PubMed for relevant papers
- Processes multiple abstracts
- Aggregates and deduplicates relations
- Useful for comprehensive literature review

## Important Implementation Notes

### Model Loading
- Models are loaded lazily on first use (singleton pattern in ner_processor.py)
- BioBERT auto-downloads from Hugging Face on first run
- scispaCy models must be manually installed (see setup commands)
- GPU automatically detected via `torch.device()`

### Error Handling
- PubMed API errors return appropriate HTTP status codes
- Invalid PMIDs handled gracefully with error messages
- Model loading failures log installation instructions
- File upload encoding issues handled with latin-1 fallback

### Data Flow
1. User submits text/PMID via web interface or API
2. Text fetched from PubMed if PMID provided
3. NER models extract entities in parallel
4. BioBERT processes sentence-by-sentence for relations
5. Results filtered by confidence threshold
6. JSON saved to `data/results/` with timestamp
7. Structured response returned to frontend

### Performance Considerations
- BioBERT runs on GPU if available (check with `torch.cuda.is_available()`)
- Semantic extraction is slowest method (use sparingly)
- PubMed rate limits enforced via time-based throttling
- Large batches split into chunks (BATCH_SIZE = 32)
- Consider caching for repeated PubMed queries (not currently implemented)

## Known Limitations

1. CSV/Excel export endpoints are TODO (app.py:431, 434)
2. File upload only supports text files effectively (PDF/DOC reading not implemented)
3. GPU acceleration requires manual PyTorch CUDA setup
4. No database persistence - results only saved as JSON files
5. No cross-document entity resolution or knowledge graph construction
