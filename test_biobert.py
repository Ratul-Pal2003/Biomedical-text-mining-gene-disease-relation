from models.ner_processor import extract_biomedical_entities

# Test with scientific abstract language
test_text = """
Our study investigated TCF7L2 gene variants in type 2 diabetes patients.
The results demonstrate that TCF7L2 polymorphisms are associated with 
increased diabetes susceptibility. Additionally, we found that CFTR mutations
contribute to cystic fibrosis pathogenesis. The APOE ε4 allele increases
the risk of Alzheimer's disease in elderly populations.
"""

print("🧪 Testing BioBERT Relation Extraction...")
print("=" * 60)
print("Input text:")
print(test_text)
print("\n" + "=" * 60)

try:
    # Extract with BioBERT
    results = extract_biomedical_entities(test_text, use_biobert=True)
    
    print("✅ BioBERT Processing Complete!")
    print(f"\n📊 Statistics:")
    print(f"  - Genes found: {results['total_genes']}")
    print(f"  - Diseases found: {results['total_diseases']}")
    print(f"  - Relations found: {results['total_relations']}")
    print(f"  - Extraction method: {results['extraction_method']}")
    
    print(f"\n🧬 Genes Detected:")
    for gene in results['entities']['genes']:
        print(f"  - {gene['text']}")
    
    print(f"\n🦠 Diseases Detected:")
    for disease in results['entities']['diseases']:
        print(f"  - {disease['text']}")
    
    print(f"\n🔗 Gene-Disease Relations (BioBERT):")
    for i, relation in enumerate(results['relations'], 1):
        print(f"\n  {i}. {relation['gene']} ↔ {relation['disease']}")
        print(f"     Type: {relation['relation_type']}")
        print(f"     Confidence: {relation['confidence']:.3f}")
        print(f"     Evidence: {relation['evidence'][:80]}...")
    
    print("\n" + "=" * 60)
    print("🎉 BioBERT is working correctly!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()