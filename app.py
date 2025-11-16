from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_cors import CORS
import os
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from config import config
import logging
import re

# Import our processors
from models.ner_processor import extract_biomedical_entities, ner_processor
from utils.pubmed_api import fetch_pubmed_abstract, search_pubmed

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(config['development'])

# Enable CORS for API endpoints
CORS(app)

# Initialize configuration
config['development'].init_app(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read and return file content for processing
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    file_content = f.read()
            except UnicodeDecodeError:
                # Try with different encoding
                with open(filepath, 'r', encoding='latin-1') as f:
                    file_content = f.read()
            
            logger.info(f"File uploaded successfully: {filename}")
            return jsonify({
                'message': 'File uploaded successfully',
                'filename': filename,
                'filepath': filepath,
                'content': file_content[:1000] + ('...' if len(file_content) > 1000 else '')
            })
        else:
            return jsonify({'error': 'Invalid file type. Allowed: txt, pdf, doc, docx'}), 400
            
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        return jsonify({'error': 'File upload failed'}), 500

@app.route('/process_text', methods=['POST'])
def process_text():
    """Process text for gene-disease relation extraction using real NER and real PubMed"""
    try:
        data = request.get_json()
        
        # Get input data
        input_text = data.get('text', '')
        pubmed_id = data.get('pubmed_id', '')
        confidence_threshold = float(data.get('confidence_threshold', 0.7))
        
        # Validate input
        if not input_text and not pubmed_id:
            return jsonify({'error': 'No text or PubMed ID provided'}), 400
        
        # If PubMed ID provided, fetch REAL abstract
        pubmed_data = None
        if pubmed_id and not input_text:
            logger.info(f"Fetching REAL abstract for PubMed ID: {pubmed_id}")
            pubmed_data = fetch_pubmed_abstract(pubmed_id)
            
            if pubmed_data and pubmed_data.get('abstract'):
                input_text = pubmed_data['abstract']
                logger.info(f"Successfully fetched REAL abstract: {pubmed_data.get('title', 'No title')[:50]}...")
            else:
                return jsonify({'error': f'Could not fetch abstract for PubMed ID: {pubmed_id}. Please verify the ID is correct.'}), 404
        
        # Process text with real NER
        logger.info(f"Processing REAL text with NER (length: {len(input_text)} chars)")
        ner_results = extract_biomedical_entities(input_text, use_semantic=True)
        
        # Convert to our API format and apply confidence filtering
        api_results = []
        for relation in ner_results['relations']:
            if relation['confidence'] >= confidence_threshold:
                api_result = {
                    'disease': relation['disease'],
                    'gene': relation['gene'],
                    'evidence': relation['evidence'].strip(),
                    'confidence': round(relation['confidence'], 3),
                    'relation_type': relation.get('relation_type', 'related'),
                    'pubmed_id': pubmed_id or 'N/A',
                    'gene_info': relation['gene_info'],
                    'disease_info': relation['disease_info']
                }
                api_results.append(api_result)
        
        # Calculate statistics
        unique_genes = len(set(r['gene'] for r in api_results))
        unique_diseases = len(set(r['disease'] for r in api_results))
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(app.config['RESULTS_FOLDER'], f'results_{timestamp}.json')
        
        save_data = {
            'input_text': input_text,
            'pubmed_id': pubmed_id,
            'pubmed_data': pubmed_data,
            'confidence_threshold': confidence_threshold,
            'processing_time': timestamp,
            'results': api_results,
            'raw_ner_results': ner_results,
            'data_source': 'REAL_PUBMED_API' if pubmed_data else 'USER_INPUT'
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"Processed REAL text successfully: {len(api_results)} relations found")
        logger.info(f"Raw NER found: {ner_results['total_genes']} genes, {ner_results['total_diseases']} diseases")
        
        response_data = {
            'status': 'success',
            'results': api_results,
            'total_relations': len(api_results),
            'unique_genes': unique_genes,
            'unique_diseases': unique_diseases,
            'results_file': results_file,
            'data_source': 'REAL_PUBMED_API' if pubmed_data else 'USER_INPUT',
            'processing_stats': {
                'raw_genes_found': ner_results['total_genes'],
                'raw_diseases_found': ner_results['total_diseases'],
                'raw_relations_found': ner_results['total_relations'],
                'filtered_relations': len(api_results),
                'confidence_threshold': confidence_threshold
            }
        }
        
        # Add REAL PubMed metadata if available
        if pubmed_data:
            response_data['pubmed_info'] = {
                'title': pubmed_data.get('title', ''),
                'authors': pubmed_data.get('authors', []),
                'journal': pubmed_data.get('journal', ''),
                'year': pubmed_data.get('year', ''),
                'url': pubmed_data.get('url', ''),
                'is_real_data': True
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Text processing error: {str(e)}")
        return jsonify({'error': f'Text processing failed: {str(e)}'}), 500

@app.route('/debug_abstract/<pmid>')
def debug_abstract(pmid):
    """Show exactly what's in the abstract sentence by sentence - DEBUG ENDPOINT"""
    try:
        article = fetch_pubmed_abstract(pmid)
        if not article:
            return jsonify({'error': 'Could not fetch article'})
        
        abstract = article.get('abstract', '')
        
        # Extract entities
        entities = ner_processor.extract_entities(abstract)
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', abstract)
        
        sentence_analysis = []
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            genes_in_sentence = [g['text'] for g in entities['genes'] if g['text'].lower() in sentence.lower()]
            diseases_in_sentence = [d['text'] for d in entities['diseases'] if d['text'].lower() in sentence.lower()]
            
            sentence_analysis.append({
                'sentence_num': i+1,
                'sentence': sentence,
                'genes': genes_in_sentence,
                'diseases': diseases_in_sentence,
                'has_both': len(genes_in_sentence) > 0 and len(diseases_in_sentence) > 0
            })
        
        return jsonify({
            'pmid': pmid,
            'title': article.get('title', ''),
            'abstract': abstract,
            'total_genes': len(entities['genes']),
            'total_diseases': len(entities['diseases']),
            'all_genes': [g['text'] for g in entities['genes']],
            'all_diseases': [d['text'] for d in entities['diseases']],
            'sentences': sentence_analysis,
            'sentences_with_both': sum(1 for s in sentence_analysis if s['has_both']),
            'diagnosis': 'Genes and diseases found in same sentence' if sum(1 for s in sentence_analysis if s['has_both']) > 0 else 'Genes and diseases are in DIFFERENT sentences - no relations possible'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'type': type(e).__name__})

@app.route('/fetch_pubmed', methods=['POST'])
def fetch_pubmed():
    """Fetch abstract from PubMed ID using REAL API"""
    try:
        data = request.get_json()
        pubmed_id = data.get('pubmed_id', '')
        
        if not pubmed_id:
            return jsonify({'error': 'PubMed ID required'}), 400
        
        logger.info(f"Fetching REAL abstract for PubMed ID: {pubmed_id}")
        
        # Fetch from REAL PubMed API
        article_data = fetch_pubmed_abstract(pubmed_id)
        
        if not article_data:
            return jsonify({'error': f'Could not find article with PubMed ID: {pubmed_id}. Please check if the ID is valid.'}), 404
        
        if not article_data.get('abstract') or article_data['abstract'] == "No abstract available":
            return jsonify({'error': f'No abstract available for PubMed ID: {pubmed_id}. This article may not have an abstract.'}), 404
        
        logger.info(f"Successfully fetched REAL article: {article_data.get('title', 'No title')[:50]}...")
        
        return jsonify({
            'status': 'success',
            'pubmed_id': pubmed_id,
            'abstract': article_data['abstract'],
            'title': article_data.get('title', 'No title available'),
            'authors': article_data.get('authors', []),
            'journal': article_data.get('journal', 'Unknown journal'),
            'year': article_data.get('year', ''),
            'doi': article_data.get('doi', ''),
            'url': article_data.get('url', f'https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/'),
            'source': 'REAL_PUBMED_API'
        })
        
    except Exception as e:
        logger.error(f"PubMed fetch error: {str(e)}")
        return jsonify({'error': f'Failed to fetch from PubMed: {str(e)}'}), 500

@app.route('/search_pubmed', methods=['POST'])
def search_pubmed_endpoint():
    """Search PubMed for articles"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        max_results = int(data.get('max_results', 10))
        
        if not query:
            return jsonify({'error': 'Search query required'}), 400
        
        logger.info(f"Searching PubMed for: {query}")
        
        # Search PubMed
        articles = search_pubmed(query, max_results)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'total_results': len(articles),
            'articles': articles
        })
        
    except Exception as e:
        logger.error(f"PubMed search error: {str(e)}")
        return jsonify({'error': f'PubMed search failed: {str(e)}'}), 500
    
@app.route('/search_by_entity', methods=['POST'])
def search_by_entity():
    """
    Search for gene-disease relations by entering a gene OR disease name
    Searches PubMed, processes multiple papers, aggregates results
    """
    try:
        data = request.get_json()
        entity = data.get('entity', '').strip()
        entity_type = data.get('entity_type', 'auto')
        max_papers = int(data.get('max_papers', 10))
        confidence_threshold = float(data.get('confidence_threshold', 0.7))
        
        if not entity:
            return jsonify({'error': 'Please provide a gene or disease name'}), 400
        
        logger.info(f"🔍 Searching for relations involving: {entity}")
        
        # Build PubMed search query based on entity type
        if entity_type == 'gene':
            search_query = f'"{entity}"[Title/Abstract] AND (disease OR cancer OR syndrome)'
        elif entity_type == 'disease':
            search_query = f'"{entity}"[Title/Abstract] AND (gene OR mutation OR variant)'
        else:
            search_query = f'"{entity}"[Title/Abstract] AND (gene OR disease OR mutation)'
        
        # Search PubMed
        from utils.pubmed_api import pubmed_api
        pmids = pubmed_api.search_articles(search_query, max_papers)
        
        if not pmids:
            return jsonify({
                'status': 'no_results',
                'message': f'No papers found for "{entity}"',
                'entity': entity,
                'relations': []
            })
        
        logger.info(f"📚 Found {len(pmids)} papers, processing abstracts...")
        
        # Process each paper
        all_relations = []
        papers_processed = 0
        
        for pmid in pmids:
            try:
                article = fetch_pubmed_abstract(pmid)
                if not article or not article.get('abstract'):
                    continue
                
                abstract = article['abstract']
                papers_processed += 1
                
                # Extract relations
                results = extract_biomedical_entities(abstract, use_biobert=True)
                
                # Filter relations involving the search entity
                for relation in results['relations']:
                    entity_lower = entity.lower()
                    if (entity_lower in relation['gene'].lower() or 
                        entity_lower in relation['disease'].lower()):
                        
                        if relation['confidence'] >= confidence_threshold:
                            relation['pmid'] = pmid
                            relation['paper_title'] = article.get('title', '')
                            relation['paper_year'] = article.get('year', '')
                            relation['paper_url'] = article.get('url', '')
                            all_relations.append(relation)
                            
            except Exception as e:
                logger.error(f"Error processing {pmid}: {e}")
                continue
        
        # Remove duplicates - keep highest confidence
        unique_relations = {}
        for rel in all_relations:
            key = (rel['gene'].lower(), rel['disease'].lower())
            if key not in unique_relations or rel['confidence'] > unique_relations[key]['confidence']:
                if key not in unique_relations:
                    unique_relations[key] = rel
                    unique_relations[key]['supporting_papers'] = []
                
                unique_relations[key]['supporting_papers'].append({
                    'pmid': rel['pmid'],
                    'title': rel['paper_title'],
                    'evidence': rel['evidence']
                })
        
        final_relations = list(unique_relations.values())
        final_relations.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"✅ Found {len(final_relations)} unique relations from {papers_processed} papers")
        
        return jsonify({
            'status': 'success',
            'entity': entity,
            'papers_searched': len(pmids),
            'papers_processed': papers_processed,
            'total_relations': len(final_relations),
            'relations': final_relations
        })
        
    except Exception as e:
        logger.error(f"Search by entity error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/export/<format_type>')
def export_results(format_type):
    """Export results in specified format"""
    try:
        # Get latest results file
        results_dir = app.config['RESULTS_FOLDER']
        
        if not os.path.exists(results_dir):
            return jsonify({'error': 'No results directory found'}), 404
            
        results_files = [f for f in os.listdir(results_dir) if f.startswith('results_')]
        
        if not results_files:
            return jsonify({'error': 'No results to export'}), 404
        
        latest_file = max(results_files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
        
        with open(os.path.join(results_dir, latest_file), 'r') as f:
            full_data = json.load(f)
            results = full_data.get('results', [])
        
        if format_type == 'csv':
            # TODO: Implement CSV export
            return jsonify({'message': 'CSV export coming soon', 'data': results})
        elif format_type == 'excel':
            # TODO: Implement Excel export
            return jsonify({'message': 'Excel export coming soon', 'data': results})
        elif format_type == 'json':
            return jsonify(results)
        else:
            return jsonify({'error': 'Invalid export format'}), 400
            
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        return jsonify({'error': 'Export failed'}), 500

@app.route('/test_ner')
def test_ner_endpoint():
    """Test endpoint to verify NER is working"""
    test_text = "BRCA1 mutations cause breast cancer. APOE variants increase Alzheimer's disease risk."
    try:
        results = extract_biomedical_entities(test_text)
        return jsonify({
            'status': 'success',
            'test_text': test_text,
            'results': results
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/test_pubmed')
def test_pubmed_endpoint():
    """Test endpoint to verify PubMed API is working"""
    test_pmid = "25741868"
    try:
        article = fetch_pubmed_abstract(test_pmid)
        if article:
            return jsonify({
                'status': 'success',
                'test_pmid': test_pmid,
                'article': article
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'Could not fetch test article'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'ner_status': 'enabled',
        'biobert_status': 'enabled',
        'pubmed_api_status': 'enabled',
        'endpoints': [
            '/',
            '/upload',
            '/process_text',
            '/fetch_pubmed',
            '/search_pubmed',
            '/search_by_entity',
            '/export/<format>',
            '/test_ner',
            '/test_pubmed',
            '/debug_abstract/<pmid>',
            '/health'
        ]
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Endpoint not found'}), 404
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    
    print("🧬 BTM - Biomedical Text Mining Server")
    print("=" * 40)
    print("🚀 Starting Flask application...")
    print("📁 Upload folder:", app.config['UPLOAD_FOLDER'])
    print("📊 Results folder:", app.config['RESULTS_FOLDER'])
    print("🧠 NER: scispaCy models loaded")
    print("🤖 BioBERT: Enabled for relation extraction")
    print("📚 PubMed API: Real NCBI E-utilities integration")
    print("🌐 Server will be available at: http://localhost:5000")
    print("🧪 Test NER at: http://localhost:5000/test_ner")
    print("📄 Test PubMed at: http://localhost:5000/test_pubmed")
    print("🔍 Debug abstracts at: http://localhost:5000/debug_abstract/<pmid>")
    print("=" * 40)
    
    # Run the application
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )