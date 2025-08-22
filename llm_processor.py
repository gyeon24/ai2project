from typing import List, Dict
import re
from config import Config
from rag_chain import rag_chain, format_doc

class LLMProcessor:
    def __init__(self):
        print(f"√ Lang Chain을 위한 LLM({Config.DEFAULT_MODEL})이 준비되었습니다.")

    def gen_res(self, question: str, ps: List[Dict]) -> Dict:
        """최종 응답 생성"""
        if not ps:
            return {"answer": "관련 논문을 찾지 못해 답변을 생성할 수 없습니다.", "sources": []}
        
        format_cont = format_doc(ps)

        try:
            print("Lang Chain을 사용하여 답변 생성 중...")
            answer = rag_chain.invoke({"context": format_cont, "question": question})
        except Exception as e:
            print(f"Lang Chain 답변 생성 오류: {e}")
            answer = "답변 생성 중 오류가 발생했습니다."

        sources_info = self.prepare_sources(ps)
        return {"answer":answer.strip(), "sources": sources_info}
    
    def format_citation(self, p:Dict, n:int) -> str:
        authors = p.get('authors', [])
        author_s = "" 
        if authors:
            if len(authors) > 2:
                author_s = f"{authors[0]} et al."
            else:
                author_s = " & ".join(authors)

        year = "n.d."
        p_id = p.get('id', '')
        if 'arxiv' in p_id.lower():
            year_m = re.search(r'(\d{2})\d{2}\.', p_id)
            if year_m:
                year = f"20{year_m.group(1)}"

        title = p.get('title', '제목 없음')

        if 'arxiv' in p.get('source', '').lower():
            return f"[{n}] {author_s}. ({year}). {title} *arXiv:{p_id}*."
        
        return f"[{n}] {author_s}. ({year}). {title} *Retrieved from PubMed.*"

    def prepare_sources(self, ps:List[Dict]) -> List[Dict]:
        """출처 정보 리스트 생성"""
        return [self.format_citation(p, i) for i, p in enumerate(ps, 1)]