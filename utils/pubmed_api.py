"""
PubMed API Integration using NCBI E-utilities
Fetches abstracts and metadata from PubMed database
"""

import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
import logging
import time
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

class PubMedAPI:
    """
    PubMed API client for fetching abstracts and article metadata
    """
    
    def __init__(self, email: str = "your-email@example.com", tool: str = "BTM_GeneDisease", api_key: str = None):
        """
        Initialize PubMed API client
        
        Args:
            email (str): Your email (required by NCBI)
            tool (str): Tool name for API requests
            api_key (str): NCBI API key for higher rate limits
        """
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.email = email
        self.tool = tool
        self.api_key = api_key
        
        # Rate limiting: NCBI allows 10 requests/second with API key, 3 without
        if api_key:
            self.request_delay = 0.12  # 120ms delay between requests (8.3 req/sec to be safe)
            logger.info("PubMed API initialized with API key - higher rate limits enabled")
        else:
            self.request_delay = 0.4  # 400ms delay between requests (2.5 req/sec to be safe)
            logger.info("PubMed API initialized without API key - standard rate limits")
        
        self.last_request_time = 0
    
    def _make_request(self, endpoint: str, params: Dict) -> requests.Response:
        """
        Make rate-limited request to NCBI API
        
        Args:
            endpoint (str): API endpoint (esearch, efetch, etc.)
            params (Dict): Request parameters
            
        Returns:
            Response object
        """
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        
        # Add standard parameters
        params.update({
            'email': self.email,
            'tool': self.tool
        })
        
        # Add API key if available
        if self.api_key:
            params['api_key'] = self.api_key
        
        url = f"{self.base_url}{endpoint}.fcgi"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            self.last_request_time = time.time()
            return response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"PubMed API request failed: {e}")
            raise
    
    def search_articles(self, query: str, max_results: int = 20) -> List[str]:
        """
        Search for articles using query terms
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results
            
        Returns:
            List of PubMed IDs
        """
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'xml'
        }
        
        try:
            response = self._make_request('esearch', params)
            root = ET.fromstring(response.content)
            
            # Extract PubMed IDs from XML
            pmids = []
            for id_elem in root.findall('.//Id'):
                pmids.append(id_elem.text)
            
            logger.info(f"Found {len(pmids)} articles for query: {query}")
            return pmids
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []
    
    def fetch_article_details(self, pmid: str) -> Optional[Dict]:
        """
        Fetch detailed information for a single PubMed ID
        
        Args:
            pmid (str): PubMed ID
            
        Returns:
            Dict with article details or None if error
        """
        if not pmid or not pmid.strip():
            return None
        
        # Clean PMID (remove any non-digits)
        pmid = ''.join(filter(str.isdigit, pmid))
        if not pmid:
            logger.error("Invalid PubMed ID provided")
            return None
        
        params = {
            'db': 'pubmed',
            'id': pmid,
            'retmode': 'xml',
            'rettype': 'abstract'
        }
        
        try:
            response = self._make_request('efetch', params)
            root = ET.fromstring(response.content)
            
            # Parse XML to extract article information
            article_data = self._parse_article_xml(root, pmid)
            
            if article_data:
                logger.info(f"Successfully fetched article {pmid}: {article_data.get('title', 'No title')[:50]}...")
            else:
                logger.warning(f"No data found for PubMed ID: {pmid}")
            
            return article_data
            
        except Exception as e:
            logger.error(f"Error fetching PubMed article {pmid}: {e}")
            return None
    
    def _parse_article_xml(self, root: ET.Element, pmid: str) -> Optional[Dict]:
        """
        Parse XML response to extract article information
        
        Args:
            root (ET.Element): XML root element
            pmid (str): PubMed ID
            
        Returns:
            Dict with parsed article data
        """
        try:
            article = root.find('.//PubmedArticle')
            if article is None:
                return None
            
            # Extract basic information
            medline_citation = article.find('.//MedlineCitation')
            if medline_citation is None:
                return None
            
            # Title
            title_elem = medline_citation.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else "No title available"
            
            # Abstract
            abstract_parts = []
            abstract_section = medline_citation.find('.//Abstract')
            if abstract_section is not None:
                for abstract_text in abstract_section.findall('.//AbstractText'):
                    label = abstract_text.get('Label', '')
                    text = abstract_text.text or ''
                    if label and label.upper() not in ['BACKGROUND', 'OBJECTIVE', 'METHODS']:
                        abstract_parts.append(f"{label}: {text}" if label else text)
                    elif not label:  # Plain abstract text
                        abstract_parts.append(text)
            
            abstract = ' '.join(abstract_parts).strip()
            if not abstract:
                abstract = "No abstract available"
            
            # Authors
            authors = []
            author_list = medline_citation.find('.//AuthorList')
            if author_list is not None:
                for author in author_list.findall('.//Author'):
                    last_name = author.find('.//LastName')
                    first_name = author.find('.//ForeName')
                    if last_name is not None:
                        author_name = last_name.text
                        if first_name is not None:
                            author_name += f", {first_name.text}"
                        authors.append(author_name)
            
            # Journal
            journal_elem = medline_citation.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else "Unknown journal"
            
            # Publication date
            pub_date = medline_citation.find('.//PubDate')
            year = ""
            if pub_date is not None:
                year_elem = pub_date.find('.//Year')
                year = year_elem.text if year_elem is not None else ""
            
            # DOI
            doi = ""
            article_ids = article.find('.//ArticleIdList')
            if article_ids is not None:
                for article_id in article_ids.findall('.//ArticleId'):
                    if article_id.get('IdType') == 'doi':
                        doi = article_id.text
                        break
            
            return {
                'pmid': pmid,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'journal': journal,
                'year': year,
                'doi': doi,
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            }
            
        except Exception as e:
            logger.error(f"Error parsing XML for PMID {pmid}: {e}")
            return None
    
    def fetch_multiple_articles(self, pmids: List[str]) -> List[Dict]:
        """
        Fetch details for multiple PubMed IDs
        
        Args:
            pmids (List[str]): List of PubMed IDs
            
        Returns:
            List of article dictionaries
        """
        articles = []
        
        for pmid in pmids:
            article = self.fetch_article_details(pmid)
            if article:
                articles.append(article)
        
        return articles
    
    def search_gene_disease_articles(self, gene: str, disease: str, max_results: int = 10) -> List[Dict]:
        """
        Search for articles about specific gene-disease relationships
        
        Args:
            gene (str): Gene name
            disease (str): Disease name
            max_results (int): Maximum number of results
            
        Returns:
            List of article dictionaries
        """
        # Create search query
        query = f'("{gene}"[Title/Abstract] AND "{disease}"[Title/Abstract])'
        
        # Search for articles
        pmids = self.search_articles(query, max_results)
        
        # Fetch article details
        articles = self.fetch_multiple_articles(pmids)
        
        logger.info(f"Found {len(articles)} articles for {gene}-{disease} relationship")
        return articles

# Create global instance using config values
from config import config

# Get the configuration
app_config = config['development']

pubmed_api = PubMedAPI(
    email=app_config.PUBMED_EMAIL,
    api_key=app_config.PUBMED_API_KEY
)

def fetch_pubmed_abstract(pmid: str) -> Optional[Dict]:
    """
    Convenience function to fetch a single PubMed abstract
    
    Args:
        pmid (str): PubMed ID
        
    Returns:
        Dict with article details or None
    """
    return pubmed_api.fetch_article_details(pmid)

def search_pubmed(query: str, max_results: int = 20) -> List[Dict]:
    """
    Convenience function to search PubMed
    
    Args:
        query (str): Search query
        max_results (int): Maximum results
        
    Returns:
        List of article dictionaries
    """
    pmids = pubmed_api.search_articles(query, max_results)
    return pubmed_api.fetch_multiple_articles(pmids)