import logging, re
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

logger = logging.getLogger(__name__)

class tProcessor:
    def __init__(self):
        self.a_vec = TfidfVectorizer(max_features=1000, stop_words='english')
        self.doc_vec = None
        self.processed_doc = {}

    def ensure_data(self):
        """NLTK 데이터 확인 및 다운로드"""
        try:
            import nltk
            required = ['punkt', 'stopwords', 'wordnet']

            for d in required:
                try:
                    nltk.data.find(f'tokenizers/{d}')
                except LookupError:
                    logger.info(f"NLTK {d} 다운로드 중")
                    nltk.download(d, quiet=True)
        except ImportError:
            logger.warning("NLTK 설치되지 않음 => 기본 텍스트 처리만 사용")

    def _init_embed(self):
        try:
            from sentence_transformers import SentenceTransformer
            from config import Config

            self.embedding = SentenceTransformer(Config.EMBEDDING_MODEL)
            logging.info(f"임베딩 모델 로드 완료: {Config.EMBEDDING_MODEL}")

        except ImportError:
            logger.info("sentence-transformers 미설치 - TF-IDF만 사용")
        except Exception as e:
            logger.warning(f"임베딩 모델 로드 실패 : {e}")

    def process_doc(self, ps: List[Dict]) -> List[Dict]:
        """논문 일괄 처리"""
        self.ind_map = {}
        corpus = []
        
        for p in ps:
            full_text = p.get('full_text') or p.get('abstract', '')
            if full_text and len(full_text.strip()) > 50:
                clean_text = re.sub(r'\s+', ' ', full_text).strip()
                p['clean_text'] = clean_text
                corpus.append(clean_text)
                self.ind_map[len(corpus) -1] = p
        if corpus:
            self.doc_vec = self.a_vec.fit_transform(corpus)
        
        return ps

    def single_doc(self, p: Dict) -> Optional[Dict]:
        """단일 논문 처리"""
        id = p.get('id', 'unknown')

        full_text = p.get('full_text', '')
        abstract = p.get('abstract', '')
        
        if len(full_text) >= 200:
            main_text = full_text
            text_type = 'full_text'
        elif len(abstract) > 50:
            main_text = abstract
            text_type = 'abstract'
        else:
            logger.warning(f"논문 {id}: 처리할 텍스트가 없습니다.")
            return None
        
        c_text = self.clean(main_text)
        if len(c_text.strip()) < 100:
            logger.warning(f"논문 {id}: 정제 후 텍스트 부족")
            return None
        
        keys = self.Ex_keys(c_text)
        summary = self.gen_sum(c_text, max = 3)

        emb = None
        if self.embedding:
            try:
                emb = self.embedding.encode(c_text[:2000])
            except Exception as e :
                logger.warning(f"임베딩 생성 실패 {id}: {e}")
        
        processed_p = {
            'id': id,
            'title': p.get('title', ''),
            'authors': p.get('authors', []),
            'source' : p.get('source', ''),
            'year': p.get('year', ''),
            'original_text': main_text,
            'clean_text': c_text,
            'text_type': text_type,
            'keywords': keys,
            'summary': summary,
            'embedding': emb.tolist() if emb is not None else None,
            'text_stats': self.cal_text(c_text),
            'web_url': p.get('web_url'),
            'pdf_url': p.get('pdf_url')
        }
        
        self.processed_doc[id] = processed_p
        return processed_p

    def clean(self, text: str) -> str:
        """텍스트 정리"""
        if not text: return ""
        
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        sentence = text.split('.')
        c_sentence = []

        for s in sentence:
            s = s.strip()

            if len(s) > 10 and not s.isdigit():
                c_sentence.append(s)

        return '. '.join(c_sentence)

    def gen_sum(self, text: str, max:int = 3) -> str:
        """텍스트 요약"""
        sentence = text.split('. ')

        if len(sentence) <= max:
            return text
        
        scores = []

        for i, s in enumerate(sentence):
            score = 0

            length = len(s.split())
            if 10 <= length <= 30:
                score += 1

            if i < len(sentence) * 0.3:
                score += 0.5

            key = self.Ex_keys(s, max = 5)
            score += len(key) * 0.2

            scores.append((i, score, s))

            sel_s = sorted(scores[:max], key = lambda x: x[0])

        sum_text = '. '.join([s[2] for s in sel_s])
        return sum_text
    
    def cal_text(self, text:str) -> Dict:
        """텍스트 계산"""
        w = text.split()
        s = text.split('.')

        return {
            'char_count': len(text),
            'word_count': len(w),
            'sentence_count': len(s),
            'avg_words_per_sentence': len(w) / max(len(s), 1),
            'avg_chars_per_word': len(text) / max(len(w), 1)
        }
    
    def build_vec(self, text: List[str]):
        try:
            self.a_vec = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1,2),
                min_df=2,
                max_df=0.8
            )

            self.doc_vec = self.a_vec.fit_transform(text)
            logger.info(f"TF-IDF 벡터 구축 완료: {self.doc_vec.shape}")

        except Exception as e:
            logger.error(f"TF-IDF 벡터 구축 실패: {e}")

    def Ex_keys(self, text: str, max :int = 10) -> List[str]:
        """키워드 추출"""
        try:
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize

            stop_w = set(stopwords.words('english'))

            academic = {
                'paper', 'study', 'research', 'analysis', 'method', 'result', 
                'conclusion', 'abstract', 'introduction', 'discussion'
            }
            stop_w.update(academic)

        except ImportError:
            stop_w = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
            }

        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filter_word = [w for w in words if w not in stop_w and len(w) > 2]

        word_freq = Counter(filter_word)

        top_key = [w for w, f in word_freq.most_common(max)]

        return top_key
    
    def rel_doc(self, q: str, top_k: int = 5) -> List[Dict]:
        """질문에 가장 관련성 높은 논문 찾기"""
        if self.doc_vec is None:
            return list(self.ind_map.values())[:top_k]
        
        q_vec = self.a_vec.transform([q])
        sim = cosine_similarity(q_vec, self.doc_vec).flatten()

        top_in = sim.argsort()[::-1][:top_k]

        rel_docs = []
        for i in top_in:
            doc = self.ind_map.get(i)
            if doc:
                doc['relevance_score'] = sim[i]
                rel_docs.append(doc)

        if rel_docs:
            logger.info(f"상위 {len(rel_docs)}개 문헌 필터링 완료 (최고 점수: {rel_docs[0].get('relevance_score', 0):.4f})")
        return rel_docs
