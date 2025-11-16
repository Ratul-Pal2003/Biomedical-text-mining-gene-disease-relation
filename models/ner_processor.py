"""
Improved Biomedical Named Entity Recognition (NER) with better evidence extraction
"""

import spacy
import scispacy
from typing import List, Dict, Tuple
import logging
import re

logger = logging.getLogger(__name__)

class BiomedicalNER:
    """
    Biomedical Named Entity Recognition processor using scispaCy models
    """
    
    def __init__(self):
        """Initialize NER models"""
        self.models = {}
        self.entity_types = {
            'DISEASE': 'diseases and conditions',
            'CHEMICAL': 'chemicals and drugs', 
            'GENE': 'genes and proteins'
        }
        
        # Try to load models
        self._load_models()
    
    def _load_models(self):
        """Load scispaCy models for different entity types"""
        try:
            # Model for diseases and chemicals (BC5CDR dataset)
            logger.info("Loading BC5CDR model for diseases and chemicals...")
            self.models['bc5cdr'] = spacy.load("en_ner_bc5cdr_md")
            
            # Model for genes and proteins (BioNLP13CG dataset)
            logger.info("Loading BioNLP13CG model for genes and proteins...")
            self.models['bionlp'] = spacy.load("en_ner_bionlp13cg_md")
            
            logger.info("All NER models loaded successfully!")
            
        except OSError as e:
            logger.error(f"Failed to load NER models: {e}")
            logger.error("Please install scispaCy models using:")
            logger.error("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz")
            logger.error("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bionlp13cg_md-0.5.3.tar.gz")
            raise
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict]]:
        """
        Extract biomedical entities from text
        
        Args:
            text (str): Input text to process
            
        Returns:
            Dict containing lists of entities by type
        """
        if not text or not text.strip():
            return {'genes': [], 'diseases': [], 'chemicals': []}
        
        entities = {
            'genes': [],
            'diseases': [],
            'chemicals': []
        }
        
        try:
            # Process with BC5CDR model (diseases and chemicals)
            if 'bc5cdr' in self.models:
                doc_bc5cdr = self.models['bc5cdr'](text)
                
                for ent in doc_bc5cdr.ents:
                    entity_info = {
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': getattr(ent, 'confidence', 0.8),
                        'sentence': self._get_sentence_containing_entity(text, ent.start_char, ent.end_char)
                    }
                    
                    if ent.label_ == 'DISEASE':
                        entities['diseases'].append(entity_info)
                    elif ent.label_ == 'CHEMICAL':
                        entities['chemicals'].append(entity_info)
            
            # Process with BioNLP model (genes and proteins)
            if 'bionlp' in self.models:
                doc_bionlp = self.models['bionlp'](text)
                
                for ent in doc_bionlp.ents:
                    entity_info = {
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': getattr(ent, 'confidence', 0.8),
                        'sentence': self._get_sentence_containing_entity(text, ent.start_char, ent.end_char)
                    }
                    
                    # Map various gene/protein labels to 'genes'
                    if ent.label_ in ['GENE_OR_GENE_PRODUCT', 'PROTEIN', 'GENE']:
                        entities['genes'].append(entity_info)
            
            # Remove duplicates and sort by position
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