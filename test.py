# test_pipeline.py (최종 완성 버전)

import logging
from pprint import pprint
import traceback

# 1. 최종적으로 사용하는 모든 모듈을 임포트합니다.
try:
    from search.intent_module import Intent
    from search.paper_search import Search
    from paper_download import Download
    from text_processor import tProcessor
    from llm_processor import LLMProcessor
except ImportError as e:
    print(f"모듈 임포트 실패: {e}. 모든 파일이 올바른 위치에 있는지 확인해주세요.")
    exit()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_test(test_query: str):
    
    print(f"🚀 테스트를 시작합니다. 질문: \"{test_query}\"\n")

    try:
        # --- STEP 1: 질문 의도 분석 (키워드 및 분야 추출) ---
        print("--- STEP 1: 질문 의도 분석 중... ---")
        intent_analyzer = Intent()
        keywords = intent_analyzer.Key(test_query)
        domains = intent_analyzer.Domain(test_query)
        
        if not keywords:
            print("❌ 질문에서 키워드를 추출할 수 없습니다. 테스트를 중단합니다.")
            return
            
        print(f"✅ 추출된 키워드: {keywords}")
        print(f"✅ 분석된 분야: {domains}\n")

        # --- STEP 2: 스마트 검색 (분야에 맞춰 최적의 사이트 검색) ---
        print("--- STEP 2: 스마트 검색 실행 중... ---")
        search_engine = Search()
        search_results = search_engine.search_all(keywords)
        
        if not search_results:
            print("❌ 검색된 문헌이 없습니다. 테스트를 중단합니다.")
            return

        print(f"✅ {len(search_results)}개의 고유한 문헌을 찾았습니다.")
        print(f"➡️ 찾은 문헌 전체의 콘텐츠를 추출합니다.\n")

        # --- STEP 3: 콘텐츠 다운로드 및 추출 ---
        print("--- STEP 3: 문헌 콘텐츠 추출 중... ---")
        downloader = Download()
        downloaded_papers = [p for p in (downloader.d_and_p(paper) for paper in search_results) if p]
        
        if not downloaded_papers:
            print("❌ 문헌은 찾았지만 내용을 추출할 수 없었습니다. 테스트를 중단합니다.")
            return
            
        print(f"✅ {len(downloaded_papers)}개 문헌에서 성공적으로 콘텐츠를 추출했습니다.\n")

        # --- STEP 4: 텍스트 처리 및 관련성 재순위 (Re-ranking) ---
        print("--- STEP 4: 텍스트 처리 및 관련성 재순위 실행 중... ---")
        text_processor = tProcessor()
        text_processor.process_doc(downloaded_papers)
        relevant_papers = text_processor.rel_doc(test_query, top_k=5)

        # --- STEP 5: LangChain으로 최종 답변 생성 ---
        print("--- STEP 5: LangChain으로 최종 답변 생성 중... ---")
        llm_processor = LLMProcessor()
        final_result = llm_processor.gen_res(test_query, relevant_papers)

        if not final_result or not final_result.get("answer"):
            print("❌ 최종 답변을 생성하는 데 실패했습니다.")
            return

        # --- 최종 결과 출력 ---
        print("\n" + "="*80)
        print("✅ 테스트 완료! 최종 결과:")
        print("="*80)

        print("\n[생성된 답변]")
        print(final_result.get("answer"))
        
        print("\n[참고 문헌 (APA)]")

        for source in final_result.get("sources", []):
            print(source)
        print("\n")

    except Exception as e:
        print(f"\n❌ 테스트 중 심각한 오류가 발생했습니다: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    
    # 1.컴퓨터 뷴야 테스트
    tech_query = "트랜스포머 모델이 자연어 처리 분야에서 가지는 장점은 무엇인가?"
    run_test(tech_query)

    # 2. 의학 분야 테스트
    medical_query = "CRISPR 유전자 가위 기술의 최신 임상 적용 사례와 윤리적 문제점은?"
    run_test(medical_query)

    # 3. 융합 분야
    test_query = "의료 영상 진단을 위한 CNN 기반 인공지능 모델의 정확도"
    run_test(test_query)