#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import gc
import torch
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel, LoraConfig, get_peft_model
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ─── 1) 경로 설정 ─────────────────────────────────────────
BASE_DIR    = "/home/wcjeong/my_project/future_lab/qwen3-1.7b-base"
LORA_DIR    = "/home/wcjeong/my_project/future_lab/lora_adapter"
PDF_INDEX   = '/home/wcjeong/my_project/final_result/pdf_faiss_index'
QA_INDEX    = '/home/wcjeong/my_project/final_result/qa_faiss_index'

# ─── 2) Retrieval 임베딩 & Threshold ───────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': DEVICE}
)
QA_DIST_THRESHOLD = 0.5
PDF_TOP_K         = 3
PDF_FETCH_K       = 10
MAX_PROMPT_LEN    = 512

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()
    try:
        torch.cuda.reset_peak_memory_stats()
    except:
        pass

# TOC 제거용 정규식
toc_pattern = re.compile(r'^\s*\d+(?:\.\d+)+\s+.*\.+\s*\d+\s*$')
def clean_toc_lines(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines()
        if not toc_pattern.match(line)
    )

# FAISS 스토어 로드
qa_store = FAISS.load_local(
    folder_path=os.path.join(QA_INDEX, 'qa_index'),
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)
pdf_store = FAISS.load_local(
    folder_path=PDF_INDEX,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

# RAG 기반 Context 추출
def answer_question_general(query: str) -> str:
    best_doc, dist = qa_store.similarity_search_with_score(query, k=1)[0]
    if dist <= QA_DIST_THRESHOLD:
        return best_doc.metadata.get('context', '')
    docs = pdf_store.max_marginal_relevance_search(
        query, k=PDF_TOP_K, fetch_k=PDF_FETCH_K
    )
    cleaned = [clean_toc_lines(d.page_content) for d in docs]
    return "\n---\n".join(cleaned)

# ─── 3) 토크나이저 & 베이스 모델 로딩 ───────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    BASE_DIR,
    trust_remote_code=True
)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_DIR,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
    use_cache=True,
)

# ─── 4) PEFT-LoRA 어댑터 로딩 ─────────────────────────────
peft_model = PeftModel.from_pretrained(
    base_model,
    LORA_DIR,
    torch_dtype="auto",
    device_map="auto",
    use_safetensors=True,
    ignore_mismatched_sizes=True
)
peft_model.eval()
base_model.eval()
print("▶ Loaded PEFT config:", peft_model.peft_config)

# ─── 5) GenerationConfig 정의 ─────────────────────────────
gen_config = GenerationConfig(
    max_new_tokens=128,
    do_sample=False,  # greedy
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)

# ─── 6) 추론 함수 정의 ─────────────────────────────────────
def generate_answer(question: str) -> str:
    # FAISS로부터 context 가져오기
    context = answer_question_general(question)
    print(context)
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_LEN,
        padding="longest"
    ).to(peft_model.device)
    # 생성
    outputs = peft_model.generate(
        **inputs,
        generation_config=gen_config,max_new_tokens=50
    )
    # 디코딩
    answer = tokenizer.decode(
        outputs[0][inputs.input_ids.size(1):],
        skip_special_tokens=True
    )
    return answer.strip()

# ─── 7) Gradio 인터페이스 구성 ─────────────────────────────
iface = gr.Interface(
    fn=generate_answer,
    inputs=[gr.Textbox(lines=2, label="Question")],
    outputs=gr.Textbox(lines=3, label="Answer"),
    title="Qwen3-1.7B + LoRA RAG QA",
    description="질문만 입력하면 FAISS 기반 Retrieval을 통해 Context를 가져와 답변을 생성합니다."
)
iface.launch(share=True)
# ─── 8) 앱 실행 ────────────────────────────────────────────
if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
