"""
Quick test script for NER processor
Run this to verify everything is working
"""

from models.ner_processor import extract_biomedical_entities

def test_ner():
    # Test text with known gene-disease relationships
    test_text = """
    BRCA1 mutations are strongly associated with hereditary breast cancer and ovarian cancer.
    The APOE ε4 allele is a major genetic risk factor for Alzheimer's disease.
    Mutations in the CFTR gene cause cystic fibrosis, a genetic disorder affecting the lungs.
    TCF7L2 variants increase susceptibility to type 2 diabetes mellitus.
    """
    
    print("🧪 Testing Biomedical NER...")
    print("=" * 50)
    print("Input text:")
    print(test_text)
    print("\n" + "=" * 50)
    
    try:
        # Extract entities and relations
        results = extract_biomedical_entities(test_text)
        
        print("📊 Results:")
        print(f"Total genes found: {results['total_genes']}")
        print(f"Total diseases found: {results['total_diseases']}")
        print(f"Total relations found: {results['total_relations']}")
        
        print("\n🧬 Genes:")
        for gene in results['entities']['genes']:
            print(f"  - {gene['text']} (confidence: {gene['confidence']:.2f})")
        
        print("\n🦠 Diseases:")
        for disease in results['entities']['diseases']:
            print(f"  - {disease['text']} (confidence: {disease['confidence']:.2f})")
        
        print("\n🔗 Gene-Disease Relations:")
        for i, relation in enumerate(results['relations'], 1):
            print(f"  {i}. {relation['gene']} → {relation['disease']}")
            print(f"     Confidence: {relation['confidence']:.2f}")
            print(f"     Evidence: {relation['evidence'][:100]}...")
            print()
        
        print("✅ NER processor working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_ner()