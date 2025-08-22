from typing import List, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from config import Config

llm = ChatGoogleGenerativeAI(model=Config.DEFAULT_MODEL, temperature=0.1)

prompt_t = """
당신은 주어진 학술 문헌들을 분석하고 핵심 내용을 요약하는 전문 연구 분석가입니다.
아래 '문헌 정보'만을 근거로 하여, 사용자의 '질문'에 대한 답변을 다음 형식에 맞춰 한국어로 생성해 주세요.

[답변 형식]
## 개요
- 질문에 대한 핵심적인 내용을 2~3 문장으로 요약합니다.

## 주요 내용
- 문헌에서 찾은 주요 사실이나 근거들을 글머리 기호(bullet point)를 사용하여 항목별로 서술합니다.
- 각 항목은 **핵심 용어**를 굵게 표시하고, 문장 끝에는 반드시 [출처 1], [출처 2] 형식으로 근거 문헌을 표기합니다.

## 결론
- 전체 내용을 종합하여 한 문장으로 결론을 내립니다.

---
[문헌 정보]
{context}
---
[질문]
{question}
---
[답변]
"""

prompt = ChatPromptTemplate.from_template(prompt_t)

def format_doc(doc:List[Dict]) -> str:
    format_str = []
    for i, d in enumerate(doc, 1):
        doc_con = d.get('clean_text') or d.get('summary', '내용 없음')
        content_sni = doc_con[:3000]

        format = f"[출처 {i}]\n"
        format += f"제목: {d.get('title', '제목 없음')}\n"
        format += f"요약: {content_sni}..."

        format_str.append(format)

    return "\n\n".join(format_str)

rag_chain = (
    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)