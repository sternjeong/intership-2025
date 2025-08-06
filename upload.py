import os
import random
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 설정
PDF_DIR    = '/home/wcjeong/my_project/final_result/pdf_save'
QA_JSON    = '/home/wcjeong/my_project/final_result/dataset/TeleQuAD/extractive/v4/TeleQuAD-v4-full.json'
PDF_INDEX  = '/home/wcjeong/my_project/final_result/pdf_faiss_index'
QA_INDEX   = '/home/wcjeong/my_project/final_result/qa_faiss_index'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

# 임베딩 모델 생성
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': DEVICE}
)

# 거리(L2) 기준 임계치 설정
DIST_THRESHOLD = 0.5  # L2 거리 0.5 이내면 QA로 매칭
TOP_K          = 3    # PDF 검색 상위 k개


# -----------------------------
# 1) QA 벡터스토어 구축/로드
# -----------------------------
def load_or_create_qa_store(qa_json_path: str, embeddings: HuggingFaceEmbeddings, index_dir: str) -> FAISS:
    """
    TeleQuAD-v4-full 형식의 JSON을 파싱해 질문-문맥 페어를 추출하고,
    QA용 FAISS 인덱스가 있으면 로드, 없으면 생성 후 저장합니다.
    """
    qa_index_path = os.path.join(index_dir, "qa_index")
    os.makedirs(qa_index_path, exist_ok=True)

    # 기존 인덱스 로드
    if os.path.isdir(qa_index_path) and os.path.exists(os.path.join(qa_index_path, "index.faiss")):
        print("QA 인덱스 로드:", qa_index_path)
        return FAISS.load_local(
            folder_path=qa_index_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

    # JSON 로드
    with open(qa_json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    qa_entries = []

    # TeleQuAD 구조: data -> docs -> paragraphs -> qas
    for doc in raw.get('data', []):
        for para in doc.get('paragraphs', []):
            context = para.get('context', '')
            for qa in para.get('qas', []):
                # 질문 하나당 컨텍스트는 해당 paragraph 전체
                qa_entries.append({
                    'question': qa.get('question', ''),
                    'context': context,
                    'qa_id': qa.get('id')
                })

    questions = [e['question'] for e in qa_entries]
    contexts  = [e['context']  for e in qa_entries]

    print(f"QA 인덱스 생성: 총 {len(questions)}개의 질문")
    qa_store = FAISS.from_texts(
        texts=questions,
        metadatas=[{'context': c, 'qa_id': qa_entries[i]['qa_id']} for i, c in enumerate(contexts)],
        embedding=embeddings
    )
    qa_store.save_local(qa_index_path)
    print("QA 인덱스 저장 완료", qa_index_path)
    return qa_store

# -----------------------------
# 2) PDF 벡터스토어 구축/업데이트
# -----------------------------
def build_or_update_pdf_store(pdf_paths: list[str], index_dir: str, embeddings: HuggingFaceEmbeddings,
                               chunk_size: int = 2000, chunk_overlap: int = 400) -> FAISS:
    os.makedirs(index_dir, exist_ok=True)
    # 기존 인덱스 로드 시도
    if os.path.isdir(index_dir) and os.path.exists(os.path.join(index_dir, "index.faiss")):
        print("기존 PDF 인덱스 로드:", index_dir)
        vs = FAISS.load_local(
            folder_path=index_dir,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("새 PDF 인덱스 생성 예정:", index_dir)
        vs = None

    # PDF → 청크 변환
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[". ", "? ", "! "]
    )
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        for idx, chunk in enumerate(chunks):
            chunk.metadata['source']   = os.path.basename(pdf_path)
            chunk.metadata['chunk_id'] = f"{os.path.basename(pdf_path)}_chunk_{idx}"
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("추가할 PDF 청크가 없습니다.")

    # 색인 생성 or 업데이트
    if vs is None:
        print(f"PDF 인덱스 생성: {len(all_chunks)}개 청크")
        vs = FAISS.from_documents(all_chunks, embeddings)
    else:
        print(f"기존 인덱스에 {len(all_chunks)}개 청크 추가")
        vs.add_documents(all_chunks)

    vs.save_local(index_dir)
    print(f"PDF 인덱스 저장 완료: {index_dir} (총 청크: {len(vs.docstore._dict)})")
    return vs

# -----------------------------
# 3) 질의 응답 통합 함수
# -----------------------------
def answer_question(query: str, qa_store: FAISS, pdf_store: FAISS, top_k: int = TOP_K) -> str:
    # 1) QA DB에서 유사 질문 탐색
    results = qa_store.similarity_search_with_score(query, k=1)
    best_doc, distance = results[0]
    print(f"QA 매칭 L2 거리: {distance:.4f}")
    # 거리가 작으면(유사하면) QA 컨텍스트 반환
    if distance <= DIST_THRESHOLD:
        return best_doc.metadata['context']
    # 2) PDF 코퍼스에서 청크 탐색
    docs = pdf_store.similarity_search(query, k=top_k)
    return "\n---\n".join([d.page_content for d in docs])

# -----------------------------
# 4) 메인
# -----------------------------
if __name__ == '__main__':
    # 1) 인덱스 초기화
    qa_store   = load_or_create_qa_store(QA_JSON, embeddings, QA_INDEX)
    pdf_paths  = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    pdf_store  = build_or_update_pdf_store(pdf_paths, PDF_INDEX, embeddings)

    # 2) 사용자 질문 루프
    while True:
        q = input("질문 입력 (종료: exit): ")
        if q.strip().lower() in ['exit', 'quit']:
            print("프로그램 종료합니다.")
            break
        response = answer_question(q, qa_store, pdf_store)
        print("응답 컨텍스트:\n", response)
