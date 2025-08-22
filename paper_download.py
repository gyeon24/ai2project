import os, requests, logging, re, time, PyPDF2
from bs4 import BeautifulSoup
from typing import Dict, Optional
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class Download:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Academic-RAG-Bot/1.0 (non-commercial)'})

    def d_and_p(self, p_info: Dict) -> Optional[Dict]:
        """
        논문 다운로드 및 텍스트 파싱
        PDF > Web > Abstract
        """
        p_id = p_info.get('id', 'unknown')
        logger.info(f"'{p_id}' 콘텐츠 추출 시작")

        # pdf
        if p_info.get('pdf_url'):
            pdf_text = self.pdf_download(p_info.get('pdf_url'))
            if pdf_text and len(pdf_text.strip()) > 500:
                logger.info(f"✅ PDF에서 성공적으로 텍스트 추출 ({len(pdf_text)}자)")
                return self.build_re(p_info, pdf_text, 'pdf')
            
        # web parse
        if p_info.get('web_url'):
            web_text = self.web_parse(p_info)
            if web_text and len(web_text.strip()) > 200:
                logger.info(f"✅ 웹페이지에서 성공적으로 텍스트 추출 ({len(web_text)}자)")
                return self.build_re(p_info, web_text, 'web')            
         
        # Abstract
        abstract = p_info.get('abstract', '')
        if abstract and len(abstract.strip()) > 50:
            logger.info(f"✅ 초록 텍스트 사용 ({len(abstract)}자)")
            return self.build_re(p_info, abstract, 'abstract')   

        logger.warning(f"{p_id}에서 유의미한 텍스트를 추출하지 못했습니다.")
        return None

    def pdf_download(self, pdf_url: Dict) -> Optional[str]:
        """PDF 다운로드 및 텍스트 추출"""
        time.sleep(1.5)
        logger.info(f"PDF 다운로드 시도: {pdf_url}")

        try:
            res = self.session.get(pdf_url, stream=True, timeout=20)
            res.raise_for_status()
            
            from io import BytesIO
            pdf_file = BytesIO(res.content)

            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_parts = [page.extract_text() for page in pdf_reader.pages]

            full_text = "\n".join(filter(None, text_parts))
            return self.clean(full_text)
        except Exception as e:
            logging.error(f"PDF 처리 중 오류 발생 {pdf_url}: {e}")
            return None

    def clean(sef, text: str) -> str:
        """추출된 텍스트 정리"""
        if not text:  return ""
        
        text = re.sub(r'\s*\n\s*', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        return text.strip()
    
    def save(self, temp_path: str, p_id:str):
        """PDF 저장"""
        try:
            name = self.filename(p_id)
            path = self.papers_dir / f"{name}.pdf"

            import shutil
            shutil.copy2(temp_path, path)
            logger.info(f"PDF 저장 완료: {path}")

        except Exception as e:
            logger.warning(f"PDF 저장 실퍠: {e}")

    def web_parse(self, p_info: Dict) -> Optional[str]:
        """웹페이지에서 텍스트 추출"""
        web_url = p_info.get('web_url')
        if not web_url: return None
        time.sleep(1.5)
        logger.info(f"웹페이지 파싱 시도:{web_url}")

        try:
            res = self.session.get(web_url, timeout=20)
            res.raise_for_status()
            soup = BeautifulSoup(res.content, 'html.parser')
            
            source = p_info.get('source', '').lower()
            parser_map={
                'pubmed': self.parse_pubmed,
                'arxiv' : self.parse_arxiv,
            }
            parser_f = parser_map.get(source)

            if parser_f:
                return self.clean(parser_f(soup))
            else:
                logger.warning(f"'{source}'에 대한 전용 파서가 없습니다.")
                return None
                
        except Exception as e:
            logging.error(f"웹페이지 파싱 오류 {web_url}: {e}")
            return None
 
    def parse_pubmed(self, soup:BeautifulSoup) -> Optional[str]:
        content = soup.select_one('div.abstract-content, div#abstract')
        return content.get_text(separator='\n') if content else ''

    def parse_arxiv(self, soup:BeautifulSoup) -> Optional[str]:
        content = soup.select_one('blockquote.abstract')
        return content.get_text(separator='\n') if content else ''
    
    def build_re(self, p_info: Dict, text: str, type:str) -> Dict:
        p_info['full_text'] = text
        p_info['content_type'] = type
        p_info['text_length'] = len(text)
        p_info['summary'] = text[:500] + '...' if len(text) > 500 else text
        return p_info
