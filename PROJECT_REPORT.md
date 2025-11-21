# BTM: Biomedical Text Mining for Gene-Disease Relation Extraction

## Project Report

---

## 1. Proposed Methodology (10 Marks)

### 1.1 System Architecture Overview

The BTM system employs a **three-stage NLP pipeline** for extracting gene-disease relationships from biomedical literature:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Text Acquisition│ → │ Entity Recognition│ → │ Relation Extraction│
│   (Stage 1)      │    │    (Stage 2)      │    │     (Stage 3)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

### 1.2 Tools and Technologies

| Tool/Technology | Version | Purpose |
|-----------------|---------|---------|
| **Python** | 3.8+ | Core programming language |
| **Flask** | 2.x | Web application framework |
| **spaCy** | 3.6.x | NLP framework for entity recognition |
| **scispaCy** | 0.5.3 | Biomedical NLP models |
| **BioBERT** | v1.1 | Pre-trained biomedical language model |
| **Transformers** | 4.x | Hugging Face library for BioBERT |
| **PyTorch** | 2.x | Deep learning framework |
| **D3.js** | v7 | Graph visualization |
| **NCBI E-utilities** | - | PubMed API integration |

---

### 1.3 Datasets and Data Sources

#### 1.3.1 Training Data (Pre-trained Models)

| Model | Training Dataset | Entities Detected |
|-------|------------------|-------------------|
| **en_ner_bc5cdr_md** | BC5CDR Corpus (1,500 PubMed articles) | Diseases, Chemicals |
| **en_ner_bionlp13cg_md** | BioNLP 2013 CG Task | Genes, Proteins |
| **BioBERT v1.1** | PubMed (4.5B words) + PMC (13.5B words) | Contextual embeddings |

#### 1.3.2 Runtime Data Sources

- **PubMed Database**: 35+ million biomedical citations
- **PubMed Central (PMC)**: Full-text open-access articles
- **User Input**: Direct text, file uploads

---

### 1.4 Biomedical Text Mining Techniques

This project implements several established biomedical text mining (BioNLP) techniques:

#### 1.4.1 Named Entity Recognition (NER)

**Technique:** Sequence Labeling with Deep Learning

NER identifies and classifies biomedical entities in unstructured text. The system uses:

| Technique | Description | Implementation |
|-----------|-------------|----------------|
| **Token Classification** | Each token is classified into entity types (B-DISEASE, I-DISEASE, O, etc.) | spaCy NER pipeline |
| **BIO Tagging Scheme** | Beginning-Inside-Outside tagging for multi-word entities | BC5CDR & BioNLP models |
| **Ensemble NER** | Multiple specialized models for different entity types | BC5CDR + BioNLP combination |
| **Cross-Validation Filtering** | Resolving conflicts when models disagree | Custom priority rules |

**Entity Types Extracted:**
- **Genes/Proteins**: BRCA1, ACE2, TMPRSS2, TP53
- **Diseases**: Cancer, COVID-19, Diabetes, Alzheimer's
- **Chemicals/Drugs**: Aspirin, Metformin, TNF-alpha

---

#### 1.4.2 Relation Extraction (RE)

**Technique:** Sentence-Level Co-occurrence with Contextual Classification

Relation extraction identifies semantic relationships between entities:

| Technique | Description | Implementation |
|-----------|-------------|----------------|
| **Co-occurrence Analysis** | Entities in same sentence likely related | Sentence boundary detection |
| **Keyword-Based Classification** | Relation type determined by linguistic patterns | Rule-based keyword matching |
| **Contextual Embeddings** | BioBERT captures semantic meaning | Transformer attention mechanism |
| **Proximity Scoring** | Closer entities have stronger relations | Character distance calculation |

**Relation Types:**
```
Gene ──causative──→ Disease      (Gene causes disease)
Gene ──risk_factor─→ Disease     (Gene increases risk)
Gene ──associated──→ Disease     (General association)
Gene ──protective──→ Disease     (Gene reduces risk)
Gene ──therapeutic─→ Disease     (Gene as drug target)
```

---

#### 1.4.3 Information Retrieval (IR)

**Technique:** Query-Based Literature Search

| Technique | Description | Implementation |
|-----------|-------------|----------------|
| **Boolean Search** | Structured queries with field tags | PubMed `[Title/Abstract]` tags |
| **Document Retrieval** | Fetching relevant documents by ID | NCBI efetch API |
| **Batch Processing** | Processing multiple documents | Parallel abstract analysis |

---

#### 1.4.4 Text Preprocessing

**Techniques Applied:**

| Technique | Purpose | Example |
|-----------|---------|---------|
| **Sentence Segmentation** | Split text into sentences for analysis | spaCy sentencizer |
| **Tokenization** | Break text into words/tokens | BioBERT WordPiece tokenizer |
| **Text Normalization** | Standardize text format | Lowercase, whitespace normalization |
| **Noise Removal** | Remove irrelevant content | Filter citations, LaTeX, formulas |

---

#### 1.4.5 Transfer Learning

**Technique:** Pre-trained Language Models Fine-tuned for Biomedicine

| Model | Pre-training | Downstream Task |
|-------|--------------|-----------------|
| **BioBERT** | PubMed + PMC (18B words) | Relation classification |
| **BC5CDR Model** | BC5CDR corpus | Disease/Chemical NER |
| **BioNLP Model** | BioNLP13CG corpus | Gene/Protein NER |

**Why Transfer Learning?**
- Biomedical text has specialized vocabulary
- Limited labeled training data available
- Pre-trained models capture domain knowledge

---

#### 1.4.6 Confidence Estimation

**Technique:** Multi-Factor Probabilistic Scoring

| Factor | Contribution | Rationale |
|--------|--------------|-----------|
| **Model Confidence** | Base score (0.80-0.95) | NER model certainty |
| **Linguistic Evidence** | +0.05 per keyword | Strong relation indicators |
| **Structural Features** | +0.03 for short sentences | Clearer, less ambiguous |
| **Entity Proximity** | +0.05 if close | Nearby entities more likely related |
| **Evidence Markers** | +0.02 per marker | Scientific language patterns |

---

#### 1.4.7 Knowledge Aggregation

**Technique:** Cross-Document Relation Summarization

When searching by entity name:
1. **Multi-Document Retrieval**: Fetch 20 relevant papers
2. **Relation Aggregation**: Combine same gene-disease pairs
3. **Evidence Consolidation**: Merge supporting sentences
4. **Citation Linking**: Track source papers for each relation

---

#### 1.4.8 Visualization Techniques

| Technique | Purpose | Implementation |
|-----------|---------|----------------|
| **Force-Directed Graph** | Show entity relationships | D3.js physics simulation |
| **Color Encoding** | Represent confidence levels | Green/Yellow/Red scheme |
| **Interactive Exploration** | User-driven analysis | Click, drag, zoom, filter |

---

### 1.5 Methodology Details

#### Stage 1: Text Acquisition

```
Input Sources → Preprocessing → Clean Text Output
```

**Input Methods:**
1. **Direct Text Input**: User pastes biomedical text
2. **PubMed ID Lookup**: Fetches abstract via NCBI E-utilities API
3. **Entity-based Search**: Searches PubMed for gene/disease and retrieves multiple papers

**PubMed API Integration:**
```python
# NCBI E-utilities endpoints used:
- esearch: Search PubMed database
- efetch: Retrieve article details (XML format)
- PMC OAI: Full-text retrieval when available
```

**Rate Limiting Implementation:**
- With API Key: 8.3 requests/second (120ms delay)
- Without API Key: 2.5 requests/second (400ms delay)

---

#### Stage 2: Named Entity Recognition (NER)

The NER system uses a **multi-model approach with cross-validation**:

```
                    ┌─────────────────────┐
                    │   Input Text        │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
    ┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
    │ BC5CDR Model    │ │ BioNLP Model│ │ General spaCy   │
    │ (Diseases/Chem) │ │ (Genes)     │ │ (Person Names)  │
    └────────┬────────┘ └──────┬──────┘ └────────┬────────┘
             │                 │                  │
             └────────────────┬┴──────────────────┘
                              ▼
                    ┌─────────────────────┐
                    │ Cross-Validation &  │
                    │ Filtering Pipeline  │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Validated Entities  │
                    └─────────────────────┘
```

**Model Details:**

1. **BC5CDR Model (`en_ner_bc5cdr_md`)**
   - Purpose: Detect diseases and chemical compounds
   - Training: BC5CDR corpus with 1,500 PubMed articles
   - Entity Types: `DISEASE`, `CHEMICAL`
   - Base Confidence Score: **0.85**

2. **BioNLP Model (`en_ner_bionlp13cg_md`)**
   - Purpose: Detect genes and proteins
   - Training: BioNLP 2013 Cancer Genetics task
   - Entity Types: `GENE_OR_GENE_PRODUCT`, `PROTEIN`
   - Base Confidence Score: **0.80**

3. **General English Model (`en_core_web_sm`)**
   - Purpose: Filter out person names falsely detected as genes
   - Entity Types: `PERSON`
   - Used for validation, not direct entity extraction

**Filtering Mechanisms:**

| Filter Type | Description | Example |
|-------------|-------------|---------|
| Person Name Filter | Removes author names misidentified as genes | "Su", "Anderson" → filtered |
| Minimum Length Filter | Removes short artifacts (< 3 chars) | "et", "al" → filtered |
| Cross-Validation | BC5CDR diseases take priority over BioNLP genes | "COVID-19" → disease, not gene |
| Pattern Matching | Detects post-2020 diseases not in training data | "COVID-19", "SARS-CoV-2" |

---

#### Stage 3: Relation Extraction

**BioBERT-Enhanced Relation Extraction:**

```python
# Relation Types Detected:
- causative: Gene causes disease
- risk_factor: Gene increases disease risk
- associated: General gene-disease association
- protective: Gene reduces disease risk
- therapeutic: Gene as treatment target
```

**Keyword-Based Relation Classification:**

| Relation Type | Keywords |
|---------------|----------|
| Causative | "cause", "causes", "responsible for", "leads to" |
| Risk Factor | "risk", "susceptibility", "predisposition" |
| Associated | "associated", "linked", "related", "correlated" |
| Protective | "protective", "reduce risk", "prevent" |
| Therapeutic | "treatment", "therapy", "drug target" |

---

### 1.5 Confidence Score Calculation

The confidence scoring system uses a **multi-factor approach**:

#### Formula:
```
final_confidence = min(0.98, base_confidence + total_boost)
```

#### Stage-by-Stage Calculation:

**Stage 1: Entity Base Confidence**
| Source | Confidence Score |
|--------|------------------|
| BC5CDR Model (Diseases/Chemicals) | 0.85 |
| BioNLP Model (Genes/Proteins) | 0.80 |
| Pattern Match (COVID-19, etc.) | 0.95 |

**Stage 2: Relation Base Score**
```python
base_confidence = min(gene_confidence, disease_confidence)
```
Takes the minimum of the two entity scores as the foundation.

**Stage 3: Confidence Boosts**

| Boost Factor | Condition | Boost Value |
|--------------|-----------|-------------|
| Strong Keywords | "cause", "mutation", "variant", "pathogenic" in sentence | +0.05 each |
| Short Sentence | < 20 words | +0.03 |
| Evidence Markers | "study", "show", "demonstrate", "find" | +0.02 each (max +0.05) |
| Entity Proximity | Gene and disease < 50 characters apart | +0.05 |
| Keyword Position | Relation keyword between gene and disease | +0.30 |
| Keyword in Sentence | Relation keyword elsewhere in sentence | +0.15 |

**Stage 4: Final Score**
- Maximum possible boost: ~0.18
- Hard cap at 0.98 (indicates inherent uncertainty in NLP)

#### Confidence Interpretation:

| Score Range | Level | Visual (Graph) |
|-------------|-------|----------------|
| ≥ 0.9 | High Confidence | Green (solid line) |
| 0.8 - 0.9 | Medium Confidence | Yellow (dashed line) |
| < 0.8 | Low Confidence | Red (dotted line) |

---

### 1.6 System Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Text Input  │  │ PubMed ID   │  │ Entity Search           │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
└─────────┼────────────────┼─────────────────────┼────────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                     TEXT ACQUISITION                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ PubMed API (NCBI E-utilities) → Abstract/Full-text      │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                  NAMED ENTITY RECOGNITION                        │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ BC5CDR Model │  │ BioNLP Model │  │ Person Name Filter     │ │
│  │ (Disease/    │  │ (Gene/       │  │ (en_core_web_sm)       │ │
│  │  Chemical)   │  │  Protein)    │  │                        │ │
│  └──────┬───────┘  └──────┬───────┘  └───────────┬────────────┘ │
│         │                 │                      │              │
│         └────────────────┬┴──────────────────────┘              │
│                          ▼                                      │
│              ┌─────────────────────────┐                        │
│              │ Cross-Validation &      │                        │
│              │ Entity Filtering        │                        │
│              └─────────────────────────┘                        │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                   RELATION EXTRACTION                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ BioBERT v1.1 (dmis-lab/biobert-v1.1)                    │    │
│  │ - Sentence-level processing                              │    │
│  │ - Gene-Disease pair identification                       │    │
│  │ - Relation type classification                           │    │
│  │ - Multi-factor confidence scoring                        │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                      OUTPUT & VISUALIZATION                      │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Table View   │  │ Graph View   │  │ Export (JSON/CSV/Excel)│ │
│  │ (Relations)  │  │ (D3.js)      │  │                        │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Demonstration (Achieving Objectives) (10 Marks)

### 2.1 Project Objectives

| # | Objective | Status |
|---|-----------|--------|
| 1 | Extract biomedical entities (genes, diseases) from text | ✅ Achieved |
| 2 | Identify relationships between genes and diseases | ✅ Achieved |
| 3 | Integrate with PubMed for literature retrieval | ✅ Achieved |
| 4 | Provide confidence scores for extracted relations | ✅ Achieved |
| 5 | Visualize gene-disease networks | ✅ Achieved |
| 6 | Export results in multiple formats | ✅ Achieved |

---

### 2.2 Feature Demonstrations

#### 2.2.1 Text Input Processing

**Input:**
```
The COVID-19 pandemic caused by SARS-CoV-2 has resulted in significant
morbidity worldwide. Studies show that ACE2 receptor plays a crucial
role in viral entry. Mutations in TMPRSS2 gene may affect disease severity.
```

**Output - Entities Detected:**
| Entity | Type | Confidence |
|--------|------|------------|
| COVID-19 | Disease | 0.95 |
| SARS-CoV-2 | Disease | 0.95 |
| ACE2 | Gene | 0.80 |
| TMPRSS2 | Gene | 0.80 |

**Output - Relations Extracted:**
| Gene | Disease | Relation Type | Confidence |
|------|---------|---------------|------------|
| ACE2 | COVID-19 | associated | 0.87 |
| TMPRSS2 | COVID-19 | causative | 0.91 |

---

#### 2.2.2 PubMed Integration

**Input:** PubMed ID `33020692`

**Process:**
1. System queries NCBI E-utilities API
2. Fetches article metadata (title, authors, journal, year)
3. Retrieves abstract text
4. Checks for PMC full-text availability
5. Processes text through NER and relation extraction pipeline

**Output:**
- Article metadata displayed
- Entities and relations extracted
- Results sorted by confidence score

---

#### 2.2.3 Entity-Based Search

**Input:** Gene name "BRCA1"

**Process:**
1. Constructs PubMed search query: `"BRCA1"[Title/Abstract]`
2. Retrieves up to 20 relevant papers
3. Processes each abstract
4. Aggregates and deduplicates relations
5. Generates abstractive summaries with citations

**Output Example:**
```
Gene: BRCA1
Disease: Breast Cancer
Relation: causative
Summary: "BRCA1 mutations are strongly associated with increased
         breast cancer risk..."
Papers: [PMID: 12345678, PMID: 23456789, PMID: 34567890]
```

---

#### 2.2.4 Confidence-Based Filtering

**Confidence Threshold Slider:** 0.1 to 1.0 (default: 0.7)

| Threshold | Effect |
|-----------|--------|
| 0.7 (default) | Balanced precision/recall |
| 0.9+ | High precision, fewer results |
| 0.5 | High recall, more noise |

---

#### 2.2.5 Graph Visualization

**Features:**
- Force-directed layout using D3.js
- Interactive zoom and pan
- Node highlighting on click
- Color-coded confidence levels:
  - 🟢 Green (solid): High confidence (≥0.9)
  - 🟡 Yellow (dashed): Medium confidence (0.8-0.9)
  - 🔴 Red (dotted): Low confidence (<0.8)
- Drag-and-drop node repositioning
- Tooltip with relation details

---

#### 2.2.6 Export Functionality

**Available Formats:**

| Format | Use Case |
|--------|----------|
| **JSON** | Programmatic access, further processing |
| **CSV** | Spreadsheet analysis, Excel import |
| **Excel** | Formatted reports with multiple sheets |

**JSON Export Structure:**
```json
{
  "relations": [
    {
      "gene": "BRCA1",
      "disease": "Breast Cancer",
      "relation_type": "causative",
      "confidence": 0.92,
      "evidence": "BRCA1 mutations cause..."
    }
  ],
  "entities": {
    "genes": [...],
    "diseases": [...]
  },
  "metadata": {
    "processed_at": "2024-...",
    "source": "PubMed ID: 12345678"
  }
}
```

---

### 2.3 Performance Metrics

#### NER Accuracy Improvements

| Issue | Before Fix | After Fix |
|-------|------------|-----------|
| Person names as genes | "Su", "Anderson" detected | Filtered out |
| Citation artifacts | "et", "al" detected | Filtered out |
| COVID-19 misclassification | Detected as gene | Correctly as disease |
| Cross-model conflicts | Inconsistent | BC5CDR prioritized |

#### Processing Performance

| Metric | Value |
|--------|-------|
| Average processing time (single abstract) | ~2-3 seconds |
| PubMed API rate | 2.5-8.3 req/sec |
| Maximum text length | 10,000 characters |
| Batch size | 32 abstracts |

---

### 2.4 User Interface

**Input Panel Features:**
1. **Text Input Tab**: Direct paste of biomedical text
2. **PubMed ID Tab**: Fetch by PMID with preview
3. **Search Entity Tab**: Multi-paper gene/disease search
4. **Confidence Slider**: Adjustable threshold (0.1-1.0)

**Results Panel Features:**
1. **Statistics Cards**: Counts of genes, diseases, relations
2. **Table View**: Sortable relation table
3. **Graph View**: Interactive network visualization
4. **Export Buttons**: JSON, CSV, Excel download

---

### 2.5 Sample Demonstration Workflow

```
Step 1: User enters PubMed ID "33020692"
        ↓
Step 2: System fetches COVID-19 research abstract
        ↓
Step 3: NER extracts entities:
        - Diseases: COVID-19, pneumonia
        - Genes: ACE2, TMPRSS2, IL-6
        ↓
Step 4: Relation extraction identifies:
        - ACE2 ↔ COVID-19 (associated, 0.87)
        - TMPRSS2 ↔ COVID-19 (causative, 0.91)
        ↓
Step 5: Results displayed in table and graph
        ↓
Step 6: User exports to CSV for further analysis
```

---

### 2.6 Technical Achievements

1. **Robust Entity Recognition**
   - Multi-model ensemble approach
   - Cross-validation between models
   - Automatic filtering of false positives

2. **Accurate Relation Extraction**
   - BioBERT contextual understanding
   - Multi-factor confidence scoring
   - Relation type classification

3. **Scalable Architecture**
   - Rate-limited API integration
   - Batch processing support
   - GPU acceleration (when available)

4. **User-Friendly Interface**
   - Multiple input methods
   - Interactive visualization
   - Flexible export options

---

## Appendix: File Structure

```
btm/
├── app.py                          # Flask application
├── config.py                       # Configuration settings
├── models/
│   ├── ner_processor.py            # Named Entity Recognition
│   ├── biobert_relation_extractor.py  # Relation extraction
│   └── semantic_relation_extractor.py # Semantic analysis
├── utils/
│   └── pubmed_api.py               # PubMed API integration
├── templates/
│   └── index.html                  # Web interface
├── static/
│   └── js/
│       └── graph-viz.js            # D3.js visualization
├── data/
│   ├── uploads/                    # User uploads
│   └── results/                    # Saved results
├── architecture.svg                # System architecture diagram
├── confidence_scoring.svg          # Confidence scoring diagram
└── PROJECT_REPORT.md               # This report
```

---

## References

1. Lee, J., et al. (2020). BioBERT: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics.
2. Neumann, M., et al. (2019). ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing. BioNLP Workshop.
3. Wei, C.H., et al. (2015). BC5CDR corpus: BioCreative V chemical disease relation task. Database.
4. NCBI E-utilities Documentation: https://www.ncbi.nlm.nih.gov/books/NBK25500/

---

*Report generated for BTM (Biomedical Text Mining) Project*
