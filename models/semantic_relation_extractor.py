from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class SemanticRelationExtractor:
    def __init__(self):
        self.model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
        self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def get_context_embedding(self, text: str, entity_text: str, window_size: int = 150):
        """Get embedding for entity's context"""
        # Find entity position
        start = text.lower().find(entity_text.lower())
        if start == -1:
            return None
        
        # Extract context window
        context_start = max(0, start - window_size)
        context_end = min(len(text), start + len(entity_text) + window_size)
        context = text[context_start:context_end]
        
        # Get embedding
        inputs = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        
        return embedding
    
    def extract_relations_semantic(self, text: str, gene_entities: List[Dict], 
                                   disease_entities: List[Dict], threshold: float = 0.65):
        """Extract relations using semantic similarity"""
        relations = []
        
        for gene in gene_entities:
            gene_emb = self.get_context_embedding(text, gene['text'])
            if gene_emb is None:
                continue
            
            for disease in disease_entities:
                disease_emb = self.get_context_embedding(text, disease['text'])
                if disease_emb is None:
                    continue
                
                # Calculate cosine similarity
                similarity = F.cosine_similarity(gene_emb.unsqueeze(0), disease_emb.unsqueeze(0))
                similarity_score = similarity.item()
                
                if similarity_score >= threshold:
                    # Extract evidence
                    gene_pos = text.lower().find(gene['text'].lower())
                    disease_pos = text.lower().find(disease['text'].lower())
                    
                    if gene_pos != -1 and disease_pos != -1:
                        start = min(gene_pos, disease_pos)
                        end = max(gene_pos + len(gene['text']), disease_pos + len(disease['text']))
                        evidence = text[max(0, start-50):min(len(text), end+50)]
                        
                        relations.append({
                            'gene': gene['text'],
                            'disease': disease['text'],
                            'gene_info': gene,
                            'disease_info': disease,
                            'confidence': round(similarity_score, 3),
                            'evidence': evidence.strip(),
                            'relation_type': 'semantic_association',
                            'extraction_method': 'semantic_embedding'
                        })
        
        return relations

# Global instance
semantic_extractor = SemanticRelationExtractor()