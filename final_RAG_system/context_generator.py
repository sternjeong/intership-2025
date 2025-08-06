import os
import re
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 설정 (기존 경로 사용)
PDF_INDEX = '/home/wcjeong/my_project/final_result/pdf_faiss_index'
QA_INDEX  = '/home/wcjeong/my_project/final_result/qa_faiss_index'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

# 임베딩 모델 (인덱스 생성 시 사용된 모델과 동일)
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': DEVICE}
)

# 임계치 및 검색 설정
QA_DIST_THRESHOLD = 0.5  # QA L2 거리 임계치
PDF_TOP_K         = 3    # 최종 반환할 PDF 청크 개수
PDF_FETCH_K       = 10   # PDF에서 먼저 가져올 후보 개수

# -----------------------------
# TOC 형태 라인 제거 함수
# -----------------------------
def clean_toc_lines(text: str) -> str:
    """
    숫자.숫자 형태의 목차 라인과 점선+페이지 번호 패턴을 제거합니다.
    예: "5.6.7 Some title ...... 117"
    """
    toc_pattern = re.compile(r'^\s*\d+(?:\.\d+)+\s+.*\.+\s*\d+\s*$')
    cleaned_lines = []
    for line in text.splitlines():
        if toc_pattern.match(line):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

# -----------------------------
# 스토어 로드 함수
# -----------------------------
def load_stores():
    # QA 스토어 로드
    qa_store = FAISS.load_local(
        folder_path=os.path.join(QA_INDEX, 'qa_index'),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    print("QA 스토어 로드 완료")

    # PDF 스토어 로드
    pdf_store = FAISS.load_local(
        folder_path=PDF_INDEX,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    print("PDF 스토어 로드 완료")

    return qa_store, pdf_store

# -----------------------------
# 질문 응답 통합 함수 (일반화)
# -----------------------------
def answer_question_general(
    query: str,
    qa_store: FAISS,
    pdf_store: FAISS,
    qa_threshold: float = QA_DIST_THRESHOLD,
    top_k: int = PDF_TOP_K,
    fetch_k: int = PDF_FETCH_K
) -> str:
    # 1) QA 검색 (L2 거리 기준)
    best_doc, distance = qa_store.similarity_search_with_score(query, k=1)[0]
    print(f"[QA] L2 거리 = {distance:.4f}")
    if distance <= qa_threshold:
        return best_doc.metadata['context']

    # 2) PDF 검색 – 초기사용 후보 확보
    initial_docs = pdf_store.similarity_search(query, k=fetch_k)

    # 3) 다양성 보장 최종 선택 (MMR)
    final_docs = pdf_store.max_marginal_relevance_search(
        query,
        k=top_k,
        fetch_k=fetch_k
    )

    # 4) TOC 라인 제거 후 최종 문맥 반환
    cleaned = [clean_toc_lines(doc.page_content) for doc in final_docs]
    return "\n---\n".join(cleaned)

# -----------------------------
# 메인
# -----------------------------
if __name__ == '__main__':
    qa_store, pdf_store = load_stores()
    while True:
        q = input("질문 입력 (종료: exit): ")
        if q.strip().lower() in ['exit', 'quit']:
            print("프로그램 종료합니다.")
            break
        response = answer_question_general(q, qa_store, pdf_store)
        print("응답 컨텍스트:\n", response)
