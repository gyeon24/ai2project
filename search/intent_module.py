import re
from typing import Dict, List
import google.generativeai as genai
from config import Config

class Intent:
    def __init__(self):
        if Config.GOOGLE_API_KEY:
            genai.configure(api_key=Config.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(Config.DEFAULT_MODEL)
        else:
            self.model = None

    def Key(self, text: str) -> List[str]:
        """키워드 추출"""
        if not self.model:
            return text.split[:5]
        
        try:
            prompt = f"""
            다음 질문에서 학술 검색에 가장 중요한 핵심 키워드 3-5개를 추출해줘.
            설명 없이 키워드만 쉼표(,)로 구분해서 한 줄로 출력해줘.
            질문: "{text}"
            """
            
            response = self.model.generate_content(prompt)
            keys = [k.strip() for k in response.text.split(',') if k.strip()]
            return keys[:5]
        except Exception as e:
            print(f"키워드 추출 오류: {e}")
            return text.split()[:5]
    
    def answer_re(s, t:str) -> Dict:
        requirements = {
            'depth': None,
            'format': None,      # 예: bullet list, essay, code
            'constraints': []
        }

        t_low = t.lower()

        if any(w in t_low for w in ['brief', 'short', '간단히', '요약']):
            requirements['depth'] = 'brief'
        elif any(w in t_low for w in ['detailed', 'explain in detail', '자세히']):
            requirements['depth'] = 'detailed'

        if any(w in t_low for w in ['list', '목록', 'bullet']):
            requirements['format'] = 'list'
        elif any(w in t_low for w in ['code', 'example', '예시']):
            requirements['format'] = 'code'

        if 'no external sources' in t_low or '출처 없이' in t_low:
            requirements['constraints'].append('no_external_sources')

        return requirements        

    def Q_type(s, t: str) -> str:
        """질문 유형 분류"""
        t = t.lower()
        
        if any(word in t for word in ['what is', 'define', '정의', '개념', 'explain']):
            return 'definition'
        elif any(word in t for word in ['how to', 'how does', 'method', '방법', '어떻게']):
            return 'process'
        elif any(word in t for word in ['compare', 'difference', 'vs', '차이', '비교']):
            return 'comparison'
        elif any(word in t for word in ['why', 'reason', 'cause', '왜', '이유', '원인']):
            return 'causation'
        elif any(word in t for word in ['current', 'recent', 'trend', '현재', '최근', '동향']):
            return 'current_state'
        elif any(word in t for word in ['application', 'use', 'apply', '활용', '응용', '사용']):
            return 'application'
        else: return 'general'

    
        
    def simple_key(s, t: str) -> List[str]:
        """간단한 키워드 추출 (폴백)"""
        import nltk
        try:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words = re.findall(r'\b\w{3,}\b', t.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return list(dict.fromkeys(keywords))[:5]
    
    
    def Domain(self, text: str) -> List[str]:
        """학술 영역 식별"""
        domain_keywords = {
            'medicine': [
                'medical', 'healthcare', 'diagnosis', 'treatment', 'clinical', 'disease',
                '의료', '의학', '진단', '치료', '임상', '질병', '환자', '병원'
            ],
            'biology': [
                'biology', 'genetics', 'molecular', 'cell', 'protein', 'dna', 'rna',
                '생물학', '유전', '세포', '단백질', '바이오', '분자'
            ],
            'computer_science': [
                'ai', 'machine learning', 'deep learning', 'algorithm', 'network', 'data',
                '인공지능', '머신러닝', '딥러닝', '알고리즘', 'computing', '컴퓨터', '소프트웨어', '데이터'
            ],
            'physics': [
                'physics', 'quantum', 'particle', 'semiconductor', 'mechanics',
                '물리학', '물리', '양자', '역학', '반도체', '입자'
            ],
            'chemistry': [
                'chemistry', 'chemical', 'reaction', 'compound', 'material',
                '화학', '반응', '화합물', '소재', '분석'
            ],
            'psychology': [
                'psychology', 'cognitive', 'behavior', 'mental', 'counseling',
                '심리', '인지', '행동', '정신', '상담', '감정'
            ],
            'education': [
                'education', 'learning', 'teaching', 'pedagogy', 'student',
                '교육', '학습', '교수', '학생', '학교'
            ],
            'economics_finance': [
                'economic', 'finance', 'market', 'investment', 'gdp', 'policy',
                '경제', '금융', '시장', '투자', '부동산', '금리', '정책', '주식'
            ]
        }

        text_low = text.lower()
        domains = [d for d, key in domain_keywords.items() if any(k in text_low for k in key)]
      
        return list(dict.fromkeys(domains)) if domains else ['general']

    def Language(s, t: str) -> str:
        """언어 감지"""
        korean = len(re.findall(r'[가-힣]', t))
        english = len(re.findall(r'[a-zA-Z]', t))
        
        return 'korean' if korean > english else 'english'