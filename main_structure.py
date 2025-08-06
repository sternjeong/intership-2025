import math
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
import torch.nn.functional as F
from transformers.integrations import WandbCallback
from peft import LoraConfig, get_peft_model
import wandb
from collections import defaultdict
import evaluate
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 최대 토큰 길이 설정
MAX_LENGTH = 480

def main():
    wandb.init(project='just_test', name='1st')

    # 데이터 로드
    ds = load_dataset(
        'json',
        data_files={
            'train': '/home/wcjeong/my_project/final_result/dataset/train.json',
            'validation': '/home/wcjeong/my_project/final_result/dataset/validation.json',
            'test': '/home/wcjeong/my_project/final_result/dataset/test.json'
        }
    )
    train_ds = ds['train']
    val_ds = ds['validation']
    test_ds = ds['test']

    # 제외된 항목 카운터
    orig_train_size = len(train_ds)
    orig_val_size = len(val_ds)

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-1.7B-Base", trust_remote_code=True
    )


    # 길이 필터링 함수
    def is_within_max(example):
        prompt = f"Context: {example['context']}\n\nQuestion: {example['question']}\nAnswer: "
        total_len = (
            len(tokenizer(prompt, return_tensors='pt').input_ids[0]) +
            len(tokenizer(example['answers'][0]['text'], return_tensors='pt').input_ids[0])
        )
        return total_len <= MAX_LENGTH

    train_ds = train_ds.filter(is_within_max)
    val_ds = val_ds.filter(is_within_max)

    # 5) 제외된 샘플 수 출력
    excluded_train = orig_train_size - len(train_ds)
    excluded_val = orig_val_size - len(val_ds)
    print(f"Excluded from train: {excluded_train} / {orig_train_size}")
    print(f"Excluded from validation: {excluded_val} / {orig_val_size}")


    model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-1.7B-Base",
            trust_remote_code=True,
            device_map="auto"
        )

        # 8) AdaLoRA 설정
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=['q_proj','v_proj'], #,'k_proj','o_proj','gate_proj','up_proj','down_proj'
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    # 9) GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def preprocess_lm(examples):
        all_input_ids = []
        all_attention_mask = []
        all_labels = []

        for cxt, q, answers_list in zip(
            examples['context'],
            examples['question'],
            examples['answers']            # 배치마다 answers_list: list of dict
        ):
            # 첫 번째 정답 dict에서 텍스트 가져오기
            # (answers_list[0]이 dict, ['text']이 실제 string)
            a = answers_list[0]['text']
            instruction = 'You must extract the answer solely from the provided context. Keep your response concise and no longer than two sentences.'

            # 1) base 프롬프트
            prompt = (
                f"Context: {cxt}\n\n"
                f"Instruction: {instruction}\n"
                f"Question: {q}\n"
                "Answer:"
            )
            full = prompt + " " + a

            # 2) 토크나이즈
            tok_full = tokenizer(
                full,
                truncation=True,
                padding="max_length",
                max_length=580
            )
            ids  = tok_full['input_ids']
            mask = tok_full['attention_mask']

            # 3) prompt 길이 계산
            prompt_len = len(tokenizer(
                prompt,
                add_special_tokens=True
            ).input_ids)

            # 4) labels 생성
            lab = [-100] * len(ids)
            for idx in range(prompt_len, sum(mask)):
                lab[idx] = ids[idx]

            all_input_ids.append(ids)
            all_attention_mask.append(mask)
            all_labels.append(lab)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels
        }

    # 11) Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 12) EarlyStoppingCallback 설정
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=100,
        fp16=True,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=500,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=500,
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        lr_scheduler_type="cosine",
        save_total_limit=3
    )

    # 13) Trainer 정의
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds.map(preprocess_lm, batched=True, remove_columns=train_ds.column_names),
        eval_dataset=val_ds.map(preprocess_lm, batched=True, remove_columns=val_ds.column_names),
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[early_stopping]
    )

    # 14) 모델 학습
    trainer.train()

    # (테스트 전) 드롭아웃 등 비학습 모드로 전환
    model.eval()

    OUTPUT_DIR = "./lora_adapter"  # 원하는 저장 경로로 수정
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    rouge    = evaluate.load('rouge')
    squad    = evaluate.load('squad')
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    print("\n📌 Extractive QA 샘플 5개 출력:")
    instruction = 'You must extract the answer solely from the provided context. Keep your response concise and no longer than two sentences.'
    for i, ex in enumerate(test_ds.select(range(5))):
        prompt = (
            f"Context: {ex['context']}\n\n"
            f"Instruction: {instruction}\n"
            f"Question: {ex['question']}\n"
            "Answer:"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id
        )[0]
        pred = tokenizer.decode(gen_ids[inputs.input_ids.size(1):], skip_special_tokens=True).strip()
        print("─" * 40)
        print(f"[Prompt]\n{prompt}")
        print(f"[정답] {ex['answers'][0]['text']}")
        print(f"[모델 출력] {pred}")


    predictions, references, squad_inputs = [], [], []
    for ex in test_ds:
        instruction = 'You must extract the answer solely from the provided context. Keep your response concise and no longer than two sentences.'
        prompt = (
            f"Context: {ex['context']}\n\n"
            f"Instruction: {instruction}\n"
            f"Question: {ex['question']}\n"
            "Answer:"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id
        )[0]
        pred = tokenizer.decode(gen_ids[inputs.input_ids.size(1):], skip_special_tokens=True).strip()

        predictions.append(pred)
        references.append(ex['answers'][0]['text'])
        squad_inputs.append({
            'id': str(ex['id']),
            'prediction_text': pred,
            'answers': {
                'text':         [a['text']          for a in ex['answers']],
                'answer_start': [a['answer_start']  for a in ex['answers']],
            }
        })

    # ROUGE‑1, ROUGE‑L
    rouge_res = rouge.compute(predictions=predictions, references=references)
    wandb.log({'rouge1':  rouge_res['rouge1'],
               'rougeL':  rouge_res['rougeL']})

    # SQuAD F1 / EM
    preds_only = [{'id': d['id'], 'prediction_text': d['prediction_text']} for d in squad_inputs]
    refs_only  = [{'id': d['id'], 'answers': d['answers']}              for d in squad_inputs]
    squad_res  = squad.compute(predictions=preds_only, references=refs_only)
    wandb.log({'f1':           squad_res['f1'],
               'exact_match':  squad_res['exact_match']})

    # 임베딩 코사인 유사도
    emb_p = embedder.encode(predictions, convert_to_tensor=True)
    emb_r = embedder.encode(references,  convert_to_tensor=True)
    sims  = cosine_similarity(emb_p.cpu().numpy(), emb_r.cpu().numpy()).diagonal()
    sem_score = float(np.mean(sims))
    wandb.log({'sem_score': sem_score})

    print("▶ Extractive QA Metrics:",
          f"ROUGE‑1={rouge_res['rouge1']:.4f},",
          f"ROUGE‑L={rouge_res['rougeL']:.4f},",
          f"F1={squad_res['f1']:.4f},",
          f"EM={squad_res['exact_match']:.4f},",
          f"SemScore={sem_score:.4f}")

if __name__ == "__main__":
    main()
