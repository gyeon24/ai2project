import requests, logging, time
import xml.etree.ElementTree as ET
from typing import List, Dict
from config import Config
from bs4 import BeautifulSoup
from urllib.parse import quote
import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Search:
    def __init__(self):
        self.max_results = Config.MAX_RESULTS
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Academic-RAG-Bot/1.0 (non-commercial)'
        })

        if Config.GOOGLE_API_KEY:
            genai.configure(api_key=Config.GOOGLE_API_KEY)
            self.gemini_model = genai.GenerativeModel(Config.DEFAULT_MODEL)
        else:
            self.gemini_model = None

        self.search_methods = {
            'arxiv': self.search_arxiv,
            'pubmed': self.search_pubmed
        }

    def scrape(self, url:str, params: dict = None) -> BeautifulSoup:
        """내부용 스크래핑 함수"""
        try:
            time.sleep(1.5)
            logging.info(f"Scraping: {url}")
            res = self.session.get(url, params=params, timeout=30)
            res.raise_for_status()
            return BeautifulSoup(res.content, "html.parser")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to scrape {url}: {e}")
            return None

    def translate(self, keyword:List[str]) -> str:
        """Gemini를 사용하여 영어 키워드 생성"""
        if not self.gemini_model or not keyword:
            return " ".join(keyword)
        
        prompt = f"""
        다음 한국어 키워드들을 조합하여, PubMed와 ArXiv 학술 검색에 가장 효과적인 영어 검색 구문(phrase)을 만들어줘.
        가장 핵심적인 의미를 잘 나타내는 2~3 단어의 간단한 구문이 가장 좋아.
        예시: ['인공지능', '신약 개발'] -> "AI in drug discovery"
        설명은 절대 추가하지 말고, 최종 검색 구문만 딱 한 줄로 출력해줘.
        한국어 키워드: {', '.join(keyword)}
        """
        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.1)
            )
            query = response.text.strip().replace('"', '')
            logging.info(f"Gemini 번역 성공: {keyword} -> '{query}'")
            return query
        except Exception as e:
            logging.error(f"Gemini 번역 중 오류 발생: {e}")
            return " ".join(keyword)
        

    def search_all(self, keyword: List[str], max_results: int = None) -> List[Dict]:
        """모든 API에서 논문 검색"""
        if not keyword: return []
        max_results = max_results or self.max_results

        q = self.translate(keyword)

        all = []
        for name, search_func in self.search_methods.items():
            try:
                ps = search_func(q, max_results // len(self.search_methods))
                logging.info(f"{name}에서 {len(ps)}개 논문 발견")
                all.extend(ps)
            except Exception as e:
                logging.error(f"'{name}' 검색 중 오류 발생: {e}")
            
        unique = list({p['title'].strip().lower(): p for p in all}.values())
        logging.info(f"총 {len(all)}개 발견, 중복 제거 후 {len(unique)}개")
        return unique[:max_results]

    def search_pubmed(s, q: str, max : int=10) -> List[Dict]:
        """PubMed API를 통한 논문 검색"""
        try:
            url = f"{Config.PUBMED_URL}esearch.fcgi"
            param = {
                "db": "pubmed",
                "term": q,
                "retmax": max,
                "retmode": "json"
            }

            res = s.session.get(url, params=param, timeout=15)
            res.raise_for_status()
            ids = res.json().get("esearchresult", {}).get("idlist", [])
            if not ids: return []
        
            f_url = f"{Config.PUBMED_URL}esummary.fcgi"
            f_params = {
                "db": "pubmed",
                "id": ','.join(ids),
                "retmode": "json"
            }

            f_res = s.session.get(f_url, params=f_params, timeout=15)
            f_res.raise_for_status()
            data = f_res.json().get("result", {})
        
            ps = []
            for pmid in ids:
                p_data = data.get(pmid)
                if not p_data:
                    continue
        
                title = p_data.get("title", "").strip()
                authors = [author.get('name', '') for author in p_data.get("authors", []) if author.get('name')]
                web_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                ps.append({
                    'id': pmid,
                    'title': title,
                    'authors': authors,
                    'abstract': p_data.get("elocationid", ""),
                    'pdf_url': None,
                    'web_url': web_url,
                    'source': 'PubMed'
                })
            return ps
        except Exception as e:
            logging.error(f"PubMed 검색 오류: {e}")
            return []

    def search_arxiv(s, q: str, max: int=10) -> List[Dict]:
        """ArXiv API를 통한 논문 검색"""
        try:
            url = f"{Config.ARXIV_URL}?search_query=all:{quote(q)}&start=0&max_results={max}"
            
            res = s.session.get(url, timeout=15)
            res.raise_for_status()
        
            root = ET.fromstring(res.text)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('atom:entry', ns)
        
            ps = []
            for e in entries:
                id = e.find('atom:id', ns).text
                p_id = id.split('/')[-1]
            
                title = e.find('atom:title', ns).text.strip()
                abstract = e.find('atom:summary', ns).text.strip()
                authors = [a.find('atom:name', ns).text for a in e.findall('atom:author', ns) if a.find('atom:name', ns) is not None]

                pdf_link = None
                for link in e.findall('atom:link', ns):
                    if link.get('title') == 'pdf':
                        pdf_link = link.get('href')
                        break

                abs_link = f"https://arxiv.org/abs/{id}"
                  
                ps.append({
                    'id': p_id,
                    'title': title,
                    'authors': authors,
                    'abstract': abstract,
                    'pdf_url': pdf_link,
                    'web_url' : abs_link,
                    'source': 'ArXiv'
                })
            return ps
        except Exception as e:
            logging.error(f"ArXiv 검색 오류: {e}")
            return []