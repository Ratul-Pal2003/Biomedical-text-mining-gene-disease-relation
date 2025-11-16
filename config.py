import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration class"""
    
    # Basic Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # File upload settings
    UPLOAD_FOLDER = 'data/uploads'
    RESULTS_FOLDER = 'data/results'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}
    
    # PubMed API settings - secure version using environment variables
    PUBMED_BASE_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
    PUBMED_EMAIL = os.environ.get('PUBMED_EMAIL', 'ratul.pal.22cse@bmu.edu.in')
    PUBMED_API_KEY = os.environ.get('PUBMED_API_KEY')  # No fallback for security

    # ML Model settings
    CONFIDENCE_THRESHOLD = 0.7
    MAX_TEXT_LENGTH = 10000
    
    # Processing settings
    BATCH_SIZE = 32
    MAX_RELATIONS_PER_REQUEST = 1000
    
    @staticmethod
    def init_app(app):
        """Initialize application with config"""
        # Create necessary directories
        Path(Config.UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
        Path(Config.RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)
        Path('data/sample_data').mkdir(parents=True, exist_ok=True)
        
        # Set Flask config
        app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
        app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}