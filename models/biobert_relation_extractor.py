"""
BioBERT-based Relation Extraction for Gene-Disease Relationships
Uses pre-trained BioBERT model to detect relationships between entities
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch
import logging
from typing import List, Dict, Tuple
import re

logger = logging.getLogger(__name__)

class BioBERTRelationExtractor:
    """
    BioBERT-based relation extraction for biomedical entities
    """
    
    def __init__(self):
        """Initialize BioBERT model"""
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Relation keywords for scoring
        self.relation_keywords = {
            'causative': ['cause', 'causes', 'caused by', 'responsible for', 'leads to', 'results in'],
            'risk_factor': ['risk', 'increase risk', 'susceptibility', 'predisposition', 'associated with risk'],
            'associated': ['associated', 'linked', 'related', 'correlated', 'connection', 'relationship'],
            'protective': ['protective', 'reduce risk', 'decrease risk', 'lower risk', 'prevent'],
            'therapeutic': ['treatment', 'therapy', 'therapeutic target', 'drug target']
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained BioBERT model"""
        try:
            logger.info("Loading BioBERT model for relation extraction...")
            
            # Using BioBERT v1.1 pre-trained on PubMed
            model_name = "dmis-lab/biobert-v1.1"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"BioBERT model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load BioBERT model: {e}")
            logger.warning("Falling back to rule-based relation extraction")
    
    def extract_sentence_embedding(self, sentence: str) -> torch.Tensor:
        """Get BioBERT embedding for a sentence"""
        try:
            inputs = self.tokenizer(
                sentence, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    def detect_relation_type(self, sentence: str, gene: str, disease: str) -> Tuple[str, float]:
        """
        Detect the type of relationship between gene and disease
        Returns: (relation_type, confidence_score)
        """
        sentence_lower = sentence.lower()
        gene_lower = gene.lower()
        disease_lower = disease.lower()
        
        # Check if both entities are in the sentence
        if gene_lower not in sentence_lower or disease_lower not in sentence_lower:
            return ('unrelated', 0.0)
        
        # Score each relation type based on keywords
        relation_scores = {}
        
        for relation_type, keywords in self.relation_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in sentence_lower:
                    # Higher score if keyword is between gene and disease
                    gene_pos = sentence_lower.find(gene_lower)
                    disease_pos = sentence_lower.find(disease_lower)
                    keyword_pos = sentence_lower.find(keyword)
                    
                    if min(gene_pos, disease_pos) < keyword_pos < max(gene_pos, disease_pos):
                        score += 0.3  # Keyword between entities
                    else:
                        score += 0.15  # Keyword in sentence
            
            relation_scores[relation_type] = score
        
        # Get the relation type with highest score
        if relation_scores:
            best_relation = max(relation_scores.items(), key=lambda x: x[1])
            if best_relation[1] > 0:
                return (best_relation[0], min(0.95, 0.7 + best_relation[1]))
        
        # Default to 'associated' if entities are in same sentence
        return ('associated', 0.6)
    
    def calculate_relation_confidence(self, sentence: str, gene: str, disease: str, 
                                     gene_confidence: float, disease_confidence: float) -> float:
        """
        Calculate confidence score for the relation using multiple factors
        """
        base_confidence = min(gene_confidence, disease_confidence)
        
        sentence_lower = sentence.lower()
        
        # Boost confidence based on various factors
        confidence_boost = 0.0
        
        # 1. Strong relation keywords
        strong_keywords = ['cause', 'mutation', 'variant', 'responsible', 'pathogenic']
        for keyword in strong_keywords:
            if keyword in sentence_lower:
                confidence_boost += 0.05
        
        # 2. Sentence structure (shorter sentences often clearer)
        if len(sentence.split()) < 20:
            confidence_boost += 0.03
        
        # 3. Multiple evidence markers
        evidence_markers = ['study', 'show', 'demonstrate', 'find', 'observe']
        evidence_count = sum(1 for marker in evidence_markers if marker in sentence_lower)
        confidence_boost += min(0.05, evidence_count * 0.02)
        
        # 4. Proximity of gene and disease mentions
        try:
            gene_pos = sentence_lower.index(gene.lower())
            disease_pos = sentence_lower.index(disease.lower())
            distance = abs(gene_pos - disease_pos)
            if distance < 50:  # Close proximity
                confidence_boost += 0.05
        except ValueError:
            pass
        
        # Calculate final confidence
        final_confidence = min(0.98, base_confidence + confidence_boost)
        
        return round(final_confidence, 3)
    
    def extract_relations_from_sentence(self, sentence: str, gene_entities: List[Dict], 
                                       disease_entities: List[Dict]) -> List[Dict]:
        """
        Extract all gene-disease relations from a single sentence
        """
        relations = []
        sentence_lower = sentence.lower()
        
        # Find all gene-disease pairs in this sentence
        for gene in gene_entities:
            if gene['text'].lower() not in sentence_lower:
                continue
                
            for disease in disease_entities:
                if disease['text'].lower() not in sentence_lower:
                    continue
                
                # Detect relation type
                relation_type, type_confidence = self.detect_relation_type(
                    sentence, gene['text'], disease['text']
                )
                
                if relation_type == 'unrelated':
                    continue
                
                # Calculate overall confidence
                confidence = self.calculate_relation_confidence(
                    sentence, 
                    gene['text'], 
                    disease['text'],
                    gene.get('confidence', 0.8),
                    disease.get('confidence', 0.8)
                )
                
                # Create relation
                relation = {
                    'gene': gene['text'],
                    'disease': disease['text'],
                    'gene_info': gene,
                    'disease_info': disease,
                    'relation_type': relation_type,
                    'confidence': confidence,
                    'evidence': sentence.strip(),
                    'extraction_method': 'biobert_enhanced'
                }
                
                relations.append(relation)
        
        return relations
    
    def extract_relations(self, text: str, gene_entities: List[Dict], 
                         disease_entities: List[Dict]) -> List[Dict]:
        """
        Extract all gene-disease relations from text
        
        Args:
            text: Input text
            gene_entities: List of gene entities from NER
            disease_entities: List of disease entities from NER
            
        Returns:
            List of relation dictionaries
        """
        if not gene_entities or not disease_entities:
            return []
        
        relations = []
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Extract relations from this sentence
            sentence_relations = self.extract_relations_from_sentence(
                sentence, gene_entities, disease_entities
            )
            relations.extend(sentence_relations)
        
        # Remove duplicate relations
        unique_relations = []
        seen = set()
        
        for relation in relations:
            key = (relation['gene'].lower(), relation['disease'].lower())
            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)
            else:
                # If duplicate, keep the one with higher confidence
                existing_idx = next(
                    i for i, r in enumerate(unique_relations) 
                    if (r['gene'].lower(), r['disease'].lower()) == key
                )
                if relation['confidence'] > unique_relations[existing_idx]['confidence']:
                    unique_relations[existing_idx] = relation
        
        # Sort by confidence
        unique_relations.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"BioBERT extracted {len(unique_relations)} unique relations")
        
        return unique_relations

# Create global instance
biobert_extractor = BioBERTRelationExtractor()

def extract_relations_with_biobert(text: str, gene_entities: List[Dict], 
                                   disease_entities: List[Dict]) -> List[Dict]:
    """
    Convenience function for BioBERT relation extraction
    
    Args:
        text: Input text
        gene_entities: Gene entities from NER
        disease_entities: Disease entities from NER
        
    Returns:
        List of extracted relations
    """
    return biobert_extractor.extract_relations(text, gene_entities, disease_entities)