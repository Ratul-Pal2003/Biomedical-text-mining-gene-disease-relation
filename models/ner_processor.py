"""
Improved Biomedical Named Entity Recognition (NER) with better evidence extraction
Uses Person Name detection and Cross-validation between models for accuracy
"""

import spacy
import scispacy
from typing import List, Dict, Tuple, Set
import logging
import re

logger = logging.getLogger(__name__)

class BiomedicalNER:
    """
    Biomedical Named Entity Recognition processor using scispaCy models
    With Person Name filtering and Cross-validation
    """

    def __init__(self):
        """Initialize NER models"""
        self.models = {}
        self.entity_types = {
            'DISEASE': 'diseases and conditions',
            'CHEMICAL': 'chemicals and drugs',
            'GENE': 'genes and proteins'
        }

        # Minimum length for valid entities (filters "et", "su", "al", etc.)
        self.min_entity_length = 3

        # Known diseases that older models (pre-2020) might misclassify as genes
        # These are disease names with patterns similar to gene nomenclature
        self.known_disease_patterns = {
            'covid-19', 'covid19', 'sars-cov-2', 'sars-cov2', 'sarscov2',
            'mers', 'mers-cov', 'h1n1', 'h5n1', 'zika', 'ebola'
        }

        # Try to load models
        self._load_models()

    def _load_models(self):
        """Load scispaCy models and general spaCy model for person detection"""
        try:
            # Model for diseases and chemicals (BC5CDR dataset)
            logger.info("Loading BC5CDR model for diseases and chemicals...")
            self.models['bc5cdr'] = spacy.load("en_ner_bc5cdr_md")

            # Model for genes and proteins (BioNLP13CG dataset)
            logger.info("Loading BioNLP13CG model for genes and proteins...")
            self.models['bionlp'] = spacy.load("en_ner_bionlp13cg_md")

            # General English model for PERSON detection
            logger.info("Loading general English model for person name detection...")
            try:
                self.models['general'] = spacy.load("en_core_web_sm")
                logger.info("General English model loaded for person filtering")
            except OSError:
                logger.warning("en_core_web_sm not found. Install with: python -m spacy download en_core_web_sm")
                logger.warning("Person name filtering will be limited")
                self.models['general'] = None

            logger.info("All NER models loaded successfully!")

        except OSError as e:
            logger.error(f"Failed to load NER models: {e}")
            logger.error("Please install scispaCy models using:")
            logger.error("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz")
            logger.error("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bionlp13cg_md-0.5.3.tar.gz")
            raise

    def _extract_person_names(self, text: str) -> Set[str]:
        """
        Extract person names from text using general spaCy model

        Args:
            text: Input text

        Returns:
            Set of person names (lowercase) found in text
        """
        person_names = set()

        if self.models.get('general') is None:
            return person_names

        try:
            doc = self.models['general'](text)
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    # Add the full name and individual parts
                    person_names.add(ent.text.lower())
                    # Also add individual name parts
                    for part in ent.text.split():
                        if len(part) > 2:  # Skip initials
                            person_names.add(part.lower())

            logger.debug(f"Found {len(person_names)} person names to filter")
            return person_names

        except Exception as e:
            logger.error(f"Error extracting person names: {e}")
            return person_names

    def _is_valid_entity(self, text: str, person_names: Set[str]) -> bool:
        """
        Validate if an entity is valid (not a person name or artifact)

        Args:
            text: Entity text
            person_names: Set of known person names from the text

        Returns:
            bool: True if valid entity, False otherwise
        """
        text_lower = text.lower().strip()
        text_clean = re.sub(r'[^a-zA-Z0-9]', '', text_lower)

        # Filter by minimum length (removes "et", "su", "al", etc.)
        if len(text_clean) < self.min_entity_length:
            logger.debug(f"Filtered '{text}' - too short")
            return False

        # Filter out person names detected by spaCy
        if text_lower in person_names:
            logger.debug(f"Filtered '{text}' - detected as person name")
            return False

        # Check individual words against person names
        for word in text.split():
            if word.lower() in person_names and len(word) > 2:
                logger.debug(f"Filtered '{text}' - contains person name '{word}'")
                return False

        # Filter citation patterns: "(Author, Year)" or "Author et al."
        if re.search(r'et\s+al', text_lower):
            logger.debug(f"Filtered '{text}' - citation pattern (et al)")
            return False

        # Filter year patterns in parentheses
        if re.match(r'.*\d{4}.*', text) and len(text_clean) < 10:
            logger.debug(f"Filtered '{text}' - contains year")
            return False

        return True

    def _extract_known_disease_patterns(self, text: str, entities: Dict, disease_texts: set):
        """
        Extract known disease patterns that may not be in model training data
        (e.g., COVID-19, SARS-CoV-2 which appeared after the models were trained)
        """
        # Patterns for diseases that post-date the training data
        disease_patterns = [
            (r'\bCOVID[-\s]?19\b', 'COVID-19'),
            (r'\bSARS[-\s]?CoV[-\s]?2\b', 'SARS-CoV-2'),
            (r'\bSARSCoV2\b', 'SARS-CoV-2'),
            (r'\bMERS[-\s]?CoV\b', 'MERS-CoV'),
            (r'\bH1N1\b', 'H1N1'),
            (r'\bH5N1\b', 'H5N1'),
            (r'\bZika\b', 'Zika'),
            (r'\bEbola\b', 'Ebola'),
            (r'\bmpox\b', 'mpox'),
            (r'\bmonkeypox\b', 'monkeypox'),
        ]

        for pattern, disease_name in disease_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matched_text = match.group(0)
                if matched_text.lower() not in disease_texts:
                    entity_info = {
                        'text': matched_text,
                        'label': 'DISEASE',
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.95,  # High confidence for pattern matches
                        'sentence': self._get_sentence_containing_entity(text, match.start(), match.end())
                    }
                    entities['diseases'].append(entity_info)
                    disease_texts.add(matched_text.lower())
                    logger.debug(f"Pattern-matched disease: '{matched_text}'")

    def extract_entities(self, text: str) -> Dict[str, List[Dict]]:
        """
        Extract biomedical entities from text with cross-validation

        Args:
            text (str): Input text to process

        Returns:
            Dict containing lists of entities by type
        """
        if not text or not text.strip():
            return {'genes': [], 'diseases': [], 'chemicals': []}

        # Step 1: Extract person names for filtering
        person_names = self._extract_person_names(text)
        logger.info(f"Detected {len(person_names)} person name tokens to filter")

        entities = {
            'genes': [],
            'diseases': [],
            'chemicals': []
        }

        # Store disease texts for cross-validation
        disease_texts = set()
        chemical_texts = set()

        try:
            # Step 1.5: Pattern-based detection for known diseases not in training data
            # (e.g., COVID-19, SARS-CoV-2 which post-date the models)
            self._extract_known_disease_patterns(text, entities, disease_texts)

            # Step 2: Process with BC5CDR model (diseases and chemicals) FIRST
            # BC5CDR is more reliable for disease detection
            if 'bc5cdr' in self.models:
                doc_bc5cdr = self.models['bc5cdr'](text)

                for ent in doc_bc5cdr.ents:
                    # Validate entity
                    if not self._is_valid_entity(ent.text, person_names):
                        continue

                    entity_info = {
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': 0.85,  # BC5CDR has good accuracy
                        'sentence': self._get_sentence_containing_entity(text, ent.start_char, ent.end_char)
                    }

                    if ent.label_ == 'DISEASE':
                        entities['diseases'].append(entity_info)
                        disease_texts.add(ent.text.lower())
                    elif ent.label_ == 'CHEMICAL':
                        entities['chemicals'].append(entity_info)
                        chemical_texts.add(ent.text.lower())

            # Step 3: Process with BioNLP model (genes and proteins)
            # Cross-validate: Don't add as gene if BC5CDR already identified as disease
            if 'bionlp' in self.models:
                doc_bionlp = self.models['bionlp'](text)

                for ent in doc_bionlp.ents:
                    # Validate entity
                    if not self._is_valid_entity(ent.text, person_names):
                        continue

                    # CROSS-VALIDATION: Skip if BC5CDR identified this as a disease
                    if ent.text.lower() in disease_texts:
                        logger.debug(f"Cross-validation: '{ent.text}' is a disease, not a gene")
                        continue

                    # CROSS-VALIDATION: Skip if BC5CDR identified this as a chemical
                    if ent.text.lower() in chemical_texts:
                        logger.debug(f"Cross-validation: '{ent.text}' is a chemical, not a gene")
                        continue

                    # Skip known disease patterns (e.g., COVID-19, SARS-CoV-2)
                    # These may be misclassified as genes by older models
                    entity_lower = ent.text.lower().replace(' ', '').replace('-', '')
                    if any(pattern.replace('-', '') in entity_lower for pattern in self.known_disease_patterns):
                        logger.debug(f"Skipping '{ent.text}' - known disease pattern")
                        # Add as disease instead if not already present
                        if ent.text.lower() not in disease_texts:
                            disease_entity = {
                                'text': ent.text,
                                'label': 'DISEASE',
                                'start': ent.start_char,
                                'end': ent.end_char,
                                'confidence': 0.90,  # High confidence for known disease
                                'sentence': self._get_sentence_containing_entity(text, ent.start_char, ent.end_char)
                            }
                            entities['diseases'].append(disease_entity)
                            disease_texts.add(ent.text.lower())
                        continue

                    entity_info = {
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': 0.80,  # Slightly lower as genes need more validation
                        'sentence': self._get_sentence_containing_entity(text, ent.start_char, ent.end_char)
                    }

                    # Map various gene/protein labels to 'genes'
                    if ent.label_ in ['GENE_OR_GENE_PRODUCT', 'PROTEIN', 'GENE']:
                        entities['genes'].append(entity_info)

            # Step 4: Remove duplicates and sort by position
            for entity_type in entities:
                entities[entity_type] = self._remove_duplicates(entities[entity_type])
                entities[entity_type].sort(key=lambda x: x['start'])

            logger.info(f"Extracted {len(entities['genes'])} genes, "
                       f"{len(entities['diseases'])} diseases, "
                       f"{len(entities['chemicals'])} chemicals")

            return entities

        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            return {'genes': [], 'diseases': [], 'chemicals': []}

    def _get_sentence_containing_entity(self, text: str, start: int, end: int) -> str:
        """Extract the sentence containing the given entity"""
        # Find sentence boundaries around the entity
        sentences = re.split(r'[.!?]+', text)

        current_pos = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_start = text.find(sentence, current_pos)
            sentence_end = sentence_start + len(sentence)

            # Check if entity is within this sentence
            if sentence_start <= start < sentence_end:
                return sentence.strip()

            current_pos = sentence_end

        # Fallback: return a window around the entity
        window_start = max(0, start - 50)
        window_end = min(len(text), end + 50)
        return text[window_start:window_end].strip()

    def _remove_duplicates(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities based on text and position"""
        seen = set()
        unique_entities = []

        for entity in entities:
            # Create a unique key based on text and position
            key = (entity['text'].lower(), entity['start'], entity['end'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return unique_entities

    def process_text_simple(self, text: str) -> Dict:
        """
        Simple processing pipeline without BioBERT

        Args:
            text (str): Input text

        Returns:
            Dict with extracted entities and relations
        """
        # Extract entities
        entities = self.extract_entities(text)

        # Simple sentence-level co-occurrence
        relations = []
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            sentence_genes = [g for g in entities['genes'] if g['text'].lower() in sentence.lower()]
            sentence_diseases = [d for d in entities['diseases'] if d['text'].lower() in sentence.lower()]

            for gene in sentence_genes:
                for disease in sentence_diseases:
                    relation = {
                        'gene': gene['text'],
                        'disease': disease['text'],
                        'gene_info': gene,
                        'disease_info': disease,
                        'confidence': min(gene['confidence'], disease['confidence']),
                        'evidence': sentence,
                        'relation_type': 'related'
                    }
                    relations.append(relation)

        # Remove duplicates
        unique_relations = []
        seen = set()
        for rel in relations:
            key = (rel['gene'].lower(), rel['disease'].lower())
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)

        return {
            'entities': entities,
            'relations': unique_relations,
            'total_genes': len(entities['genes']),
            'total_diseases': len(entities['diseases']),
            'total_relations': len(unique_relations)
        }

# Create global instance
ner_processor = BiomedicalNER()

# Import BioBERT after creating ner_processor
try:
    from models.biobert_relation_extractor import extract_relations_with_biobert
    BIOBERT_AVAILABLE = True
except ImportError:
    logger.warning("BioBERT not available, falling back to simple NER")
    BIOBERT_AVAILABLE = False

# Try to import semantic extractor
try:
    from models.semantic_relation_extractor import semantic_extractor
    SEMANTIC_AVAILABLE = True
    logger.info("Semantic relation extractor loaded successfully")
except ImportError:
    logger.warning("Semantic extractor not available")
    SEMANTIC_AVAILABLE = False

def process_text_with_biobert(text: str) -> Dict:
    """
    Enhanced processing pipeline using both NER and BioBERT

    Args:
        text (str): Input text

    Returns:
        Dict with extracted entities and BioBERT-enhanced relations
    """
    # Step 1: Extract entities using scispaCy NER
    entities = ner_processor.extract_entities(text)

    # Step 2: Use BioBERT for relation extraction
    biobert_relations = extract_relations_with_biobert(
        text,
        entities['genes'],
        entities['diseases']
    )

    # Step 3: Combine and format results
    return {
        'entities': entities,
        'relations': biobert_relations,
        'total_genes': len(entities['genes']),
        'total_diseases': len(entities['diseases']),
        'total_relations': len(biobert_relations),
        'extraction_method': 'ner_plus_biobert'
    }

def process_text_with_semantic_extraction(text: str) -> Dict:
    """
    Use semantic embeddings for relation extraction

    Args:
        text (str): Input text

    Returns:
        Dict with extracted entities and semantic relations
    """
    if not SEMANTIC_AVAILABLE:
        logger.warning("Semantic extractor not available, falling back to BioBERT")
        return process_text_with_biobert(text)

    # Step 1: Extract entities using scispaCy NER
    entities = ner_processor.extract_entities(text)

    # Step 2: Use semantic embeddings for relation extraction
    semantic_relations = semantic_extractor.extract_relations_semantic(
        text,
        entities['genes'],
        entities['diseases']
    )

    # Step 3: Combine and format results
    return {
        'entities': entities,
        'relations': semantic_relations,
        'total_genes': len(entities['genes']),
        'total_diseases': len(entities['diseases']),
        'total_relations': len(semantic_relations),
        'extraction_method': 'semantic_embedding'
    }

def extract_biomedical_entities(text: str, use_biobert: bool = True, use_semantic: bool = False) -> Dict:
    """
    Main extraction function with multiple extraction methods

    Args:
        text (str): Input text
        use_biobert (bool): Whether to use BioBERT for relations (default: True)
        use_semantic (bool): Whether to use semantic embeddings (default: False)

    Returns:
        Dict with extracted entities and relations
    """
    # Semantic extraction takes priority if requested and available
    if use_semantic and SEMANTIC_AVAILABLE:
        logger.info("Using semantic embedding for relation extraction")
        return process_text_with_semantic_extraction(text)
    elif use_biobert and BIOBERT_AVAILABLE:
        logger.info("Using BioBERT for enhanced relation extraction")
        return process_text_with_biobert(text)
    else:
        logger.info("Using simple NER-based relation extraction")
        return ner_processor.process_text_simple(text)
