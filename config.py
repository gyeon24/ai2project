import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

    PUBMED_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
    ARXIV_URL = 'http://export.arxiv.org/api/query'
    
    MAX_RESULTS = 15
    
    DEFAULT_MODEL = 'models/gemini-2.5-flash-lite'