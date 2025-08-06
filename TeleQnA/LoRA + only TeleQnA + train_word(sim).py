import math
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback
)
from transformers import default_data_collator
from transformers.integrations import WandbCallback
from peft import LoraConfig, get_peft_model
import wandb
from collections import defaultdict

# 주관식 평가 모듈
import evaluate
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def score_general(example, model, tokenizer, device):
    instruction = (
        "Among the five choices, please output the sentence or word that you believe is the correct answer. "
        "Do not output the choice index; only return the selected choice text.\n"
    )
    letters = ['0', '1', '2', '3', '4']
    prompt = (
        f"{instruction}\nQ: {example['question']}\nChoices: " +
        "  ".join(f"[{letters[i]}] {opt}" for i, opt in enumerate(example['choices'])) +
        "\nAnswer:"
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    ).to(device)
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        pad_token_id=tokenizer.eos_token_id
    )[0]
    output = tokenizer.decode(
        gen_ids[inputs.input_ids.size(1):],
        skip_special_tokens=True
    ).strip()
    def fuzzy(a, b):
        sa, sb = set(a.lower().split()), set(b.lower().split())
        return len(sa & sb) / max(1, len(sa | sb))
    scores = [fuzzy(output, opt) for opt in example['choices']]
    return int(np.argmax(scores))

class LoggingCallback(TrainerCallback):
    def __init__(self, val_dataset, tokenizer, device):
        self.val_ds = val_dataset
        self.tokenizer = tokenizer
        self.device = device

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        model = kwargs['model']
        correct, total = 0, 0
        subj_counts = defaultdict(lambda: {"correct": 0, "total": 0})

        for ex in self.val_ds:
            pred = score_general(ex, model, self.tokenizer, self.device)
            gold = ex['answer']
            subj = ex.get('subject', 'unknown')
            if pred == gold:
                correct += 1
                subj_counts[subj]["correct"] += 1
            subj_counts[subj]["total"] += 1
            total += 1

        overall_acc = correct / total
        print(f"Validation Accuracy: {overall_acc:.4f}")
        log_data = {"validation_accuracy": overall_acc}

        for subj, ct in subj_counts.items():
            acc = ct["correct"] / ct["total"]
            key = f"val_{subj.replace(' ', '_').lower()}_accuracy"
            log_data[key] = acc
            print(f"  {subj}: {acc:.4f}")

        wandb.log(log_data)


def main():
    # 1) W&B 초기화
    wandb.init(project='TeleQUAD+TeleQnA', name='lora-(64,32)-index')

    # 2) 데이터 로드 및 분할
    ds = load_dataset('netop/TeleQnA', split='test')
    split1 = ds.train_test_split(test_size=0.3, seed=42)
    train_ds, rem_ds = split1['train'], split1['test']
    split2 = rem_ds.train_test_split(test_size=1/3, seed=42)
    val_ds, test_ds = split2['train'], split2['test']

    train_ds = train_ds.shuffle(seed=42).select(range(400))  # 샘플링
    val_ds = val_ds.shuffle(seed=42).select(range(300))
    test_ds = test_ds.shuffle(seed=42).select(range(300))

    # 4) 토크나이저 & 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-1.7B-Base", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B-Base",
        trust_remote_code=True,
        device_map="auto"
    )

    # 5) TrainingArguments 정의
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=6,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-4,
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
    )

    # 6) 총 학습 스텝 수 계산
    steps_per_epoch = math.ceil(len(train_ds) / training_args.per_device_train_batch_size)


    # 8) LoRA 설정
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=[
            'q_proj','k_proj','v_proj','o_proj',
            'gate_proj','up_proj','down_proj'
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # 9) GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # 10) 전처리 함수
    def preprocess_lm(ex):
        letters=['0','1','2','3','4']
        prompts, labels = [], []
        instruction = (
            "Among the five choices, please output the sentence or word that you believe is the correct answer. "
        "Do not output the choice index; only return the selected choice text.\n\n"
        )
        for q,opts,a in zip(ex['question'], ex['choices'], ex['answer']):
            cl = [f"[{letters[i]}] {opt}" for i,opt in enumerate(opts)]
            prompts.append(f"{instruction}Q: {q}\nChoices: {'  '.join(cl)}\nAnswer:")
            labels.append(opts[a])
        return tokenizer(prompts, text_target=labels, truncation=True, padding="max_length", max_length=128)

    # 11) Data collator
    data_collator = default_data_collator

    # 12) EarlyStoppingCallback 설정
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

    # 13) Trainer 정의
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds.map(
            preprocess_lm, batched=True, remove_columns=train_ds.column_names
        ),
        eval_dataset=val_ds.map(
            preprocess_lm, batched=True, remove_columns=val_ds.column_names
        ),
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[
            WandbCallback,
            early_stopping,
            LoggingCallback(val_ds, tokenizer, device)
        ]
    )

    # 14) 모델 학습
    trainer.train()
    model.eval()

    # 15) 최종 검증 손실 및 로깅
    eval_results = trainer.evaluate()
    print(f"Final Validation Loss: {eval_results['eval_loss']:.4f}")
    wandb.log({"final_validation_loss": eval_results["eval_loss"]})

    # 16) Test accuracy 및 subject별 정확도 계산
        # 16) Test accuracy 및 subject별 정확도 계산 + 추가 메트릭
    correct, total = 0, 0
    subj_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    test_preds, test_refs = [], []

    for ex in test_ds:
        pred = score_general(ex, model, tokenizer, device)
        gold = ex['answer']
        subj = ex.get('subject', 'unknown')

        # 객관식 정답 인덱스 → 텍스트
        pred_text = ex['choices'][pred]
        gold_text = ex['choices'][gold]
        test_preds.append(pred_text)
        test_refs.append(gold_text)

        if pred == gold:
            correct += 1
            subj_counts[subj]["correct"] += 1
        subj_counts[subj]["total"] += 1
        total += 1

    # 1) Accuracy & 각 subject별
    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")
    wandb.log({"test_accuracy": test_acc})
    for subj, ct in subj_counts.items():
        acc = ct["correct"] / ct["total"]
        key = f"test_{subj.replace(' ', '_').lower()}_accuracy"
        print(f"  {subj}: {acc:.4f}")
        wandb.log({key: acc})

    # 2) Exact Match 비율
    exact_match = sum(p == r for p, r in zip(test_preds, test_refs)) / total
    wandb.log({"test_exact_match": exact_match})

    # 3) ROUGE‑1, ROUGE‑L
    rouge = evaluate.load("rouge")
    rouge_res = rouge.compute(predictions=test_preds, references=test_refs, rouge_types=["rouge1", "rougeL"])
    wandb.log({
        "test_rouge1": rouge_res["rouge1"],
        "test_rougeL": rouge_res["rougeL"]
    })

    # 4) Semantic similarity (sentence-transformers + cosine)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    emb_p = embedder.encode(test_preds, convert_to_tensor=True)
    emb_r = embedder.encode(test_refs, convert_to_tensor=True)
    sem_sims = cosine_similarity(emb_p.cpu().numpy(), emb_r.cpu().numpy()).diagonal()
    sem_score = float(sem_sims.mean())
    wandb.log({"test_semantic_score": sem_score})

    print(
        f"ExactMatch={exact_match}, "
        f"ROUGE1={rouge_res['rouge1']}, "
        f"ROUGEL={rouge_res['rougeL']}, "
        f"SemScore={sem_score:.4f}"
    )


    # ── 17) 주관식 Extractive QA 평가 (샘플 10개만) ───────────
    ext = load_dataset(
        'json',
        data_files={'test': '/home/wcjeong/my_project/TeleQuAD/extractive/v4/TeleQuAD_split/test.json'}
    )['test'].select(range(10))

    rouge    = evaluate.load('rouge')
    squad    = evaluate.load('squad')
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    predictions, references, squad_inputs = [], [], []
    for ex in ext:
        prompt = (
            f"Context: {ex['context']}\n\n"
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
