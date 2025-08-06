import math
import torch
import numpy as np
import random
from collections import defaultdict

import evaluate
import wandb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from transformers.integrations import WandbCallback
from peft import AdaLoraConfig, get_peft_model

# ───────────────────────────────────────────────
# Trainer를 수정해서 .to(device) 하지 않도록 막음
# ───────────────────────────────────────────────
class NoModelMoveTrainer(Trainer):
    def _move_model_to_device(self, model, device):
        return model

# ───────────────────────────────────────────────
# 전역 임베더 로드 (한 번만)
# ───────────────────────────────────────────────
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ───────────────────────────────────────────────
# MC 프롬프트 빌더
# ───────────────────────────────────────────────
def build_mc_prompt(question, choices):
    letters = ['1','2','3','4','5']
    instruction = (
        "Among the five choices, please output the sentence or word that you believe is the correct answer. "
        "Do not output the choice index; only return the selected choice text.\n\n"
    )
    choice_str = "  ".join(f"[{letters[i]}] {opt}" for i, opt in enumerate(choices))
    return instruction + f"Q: {question}\nChoices: {choice_str}\nAnswer:"

# ───────────────────────────────────────────────
# 객관식 평가용 점수 함수 (fuzzy + semantic)
# ───────────────────────────────────────────────
def score_general(example, model, tokenizer, device):
    # 1) 모델로부터 문장 생성
    prompt = build_mc_prompt(example['question'], example['choices'])
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
    full_output = tokenizer.decode(
        gen_ids[inputs.input_ids.size(1):],
        skip_special_tokens=True
    ).strip()

    # 2) fuzzy matching 점수
    def fuzzy_match(a, b):
        sa, sb = set(a.lower().split()), set(b.lower().split())
        return len(sa & sb) / max(1, len(sa | sb))
    fuzzy_scores = [fuzzy_match(full_output, opt) for opt in example['choices']]

    # 3) semantic similarity 점수
    out_emb = embedder.encode(full_output, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
    opt_embs = embedder.encode(example['choices'], convert_to_tensor=True).cpu().numpy()
    sem_scores = cosine_similarity(out_emb, opt_embs)[0].tolist()

    # 4) 최종 점수: fuzzy와 semantic을 50:50 가중합
    final_scores = [0.5 * f + 0.5 * s for f, s in zip(fuzzy_scores, sem_scores)]
    return int(np.argmax(final_scores))

# ───────────────────────────────────────────────
# 학습 가능한 파라미터 비율 출력
# ───────────────────────────────────────────────
def print_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"▶ Trainable params: {trainable}/{total} ({100*trainable/total:.2f}%)\n")

def main():
    wandb.init(project='TeleQUAD+TeleQnA', name='AdaLoRA-MC-Only')

    # 데이터 로드 및 분할
    ds = load_dataset('netop/TeleQnA', split='test')
    split1 = ds.train_test_split(test_size=0.3, seed=42)
    train_ds, rem_ds = split1['train'], split1['test']
    split2 = rem_ds.train_test_split(test_size=1/3, seed=42)
    val_ds, test_ds = split2['train'], split2['test']

    # 토크나이저/모델 + LoRA 설정
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base", trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B-Base",
        trust_remote_code=True,
        device_map="auto",
        use_cache=False
    )
    # AdaLoRA 설정 (예시)
    steps_per_epoch = math.ceil(len(train_ds) / 1)
    total_steps = steps_per_epoch * 5
    tinit = max(1, int(total_steps * 0.25))
    tfinal = max(1, int(total_steps * 0.2))
    adalora_cfg = AdaLoraConfig(
        init_r=64, target_r=32,
        tinit=tinit, tfinal=tfinal, total_step=total_steps,
        lora_alpha=32, lora_dropout=0.2,
        target_modules=[
            'q_proj','k_proj','v_proj','o_proj',
            'gate_proj','up_proj','down_proj'
        ],
        bias="none", task_type="CAUSAL_LM"
    )
    adalora_cfg.delta_T = 200
    model = get_peft_model(base_model, adalora_cfg)
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 학습용 전처리: 답변 텍스트만
    def preprocess_lm(examples):
        letters = ['1','2','3','4','5']
        instruction = (
            "Among the five choices, please output the sentence or word that you believe is the correct answer. "
            "Do not output the choice index; only return the selected choice text.\n\n"
        )
        prompts, targets = [], []
        for q, opts, a in zip(examples['question'], examples['choices'], examples['answer']):
            choice_fmt = [f"[{letters[i]}] {opt}" for i, opt in enumerate(opts)]
            prompts.append(instruction + f"Q: {q}\nChoices: {'  '.join(choice_fmt)}\nAnswer:")
            targets.append(opts[a])
        return tokenizer(
            prompts,
            text_target=targets,
            padding='max_length',
            truncation=True,
            max_length=198
        )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)
    training_args = TrainingArguments(
        output_dir="./results_mc_only",
        run_name="AdaLoRA-MC-Only",
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_steps=100,
        fp16=True,
        logging_steps=100,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        report_to=["wandb"],
        load_best_model_at_end=True,
        lr_scheduler_type="cosine",
        metric_for_best_model='eval_loss',
        greater_is_better=False
    )

    trainer = NoModelMoveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds.map(preprocess_lm, batched=True, remove_columns=train_ds.column_names),
        eval_dataset=val_ds.map(preprocess_lm, batched=True, remove_columns=val_ds.column_names),
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[WandbCallback(), early_stopping]
    )

    print_trainable_params(model)
    trainer.train()

    # 검증 평가
    eval_loss = trainer.evaluate()['eval_loss']
    wandb.log({"final_val_loss": eval_loss})
    print(f"Final Val Loss: {eval_loss:.4f}")

    val_counts = defaultdict(lambda: {"correct":0,"total":0})
    for ex in val_ds:
        pred_idx = score_general(ex, model, tokenizer, device)
        if pred_idx == ex['answer']:
            val_counts[ex.get('subject','unknown')]["correct"] += 1
        val_counts[ex.get('subject','unknown')]["total"] += 1
    val_total = sum(v['total'] for v in val_counts.values())
    val_acc = sum(v['correct'] for v in val_counts.values()) / val_total
    wandb.log({
        "val_overall_accuracy": val_acc,
        **{f"val_acc_{s}": c['correct']/c['total'] for s,c in val_counts.items()}
    })
    print(f"Val Acc: {val_acc:.4f}")

    # 테스트 평가 및 Rouge/EM/SemScore
    rouge_metric = load_metric('rouge')
    test_preds, test_refs = [], []
    test_counts = defaultdict(lambda: {"correct":0,"total":0})
    total, correct = 0, 0

    for ex in test_ds:
        pred_idx = score_general(ex, model, tokenizer, device)
        gold_idx = ex['answer']
        is_corr = (pred_idx == gold_idx)

        subj = ex.get('subject','unknown')
        if is_corr:
            test_counts[subj]['correct'] += 1
            correct += 1
        test_counts[subj]['total'] += 1
        total += 1

        # 기록용 텍스트
        test_preds.append(example['choices'][pred_idx])
        test_refs.append(example['choices'][gold_idx])

    test_acc = correct / total
    rouge_res = rouge_metric.compute(predictions=test_preds, references=test_refs, rouge_types=["rouge1","rouge2"])
    exact_match = sum(p==r for p,r in zip(test_preds, test_refs)) / total
    emb_p = embedder.encode(test_preds, convert_to_tensor=True)
    emb_r = embedder.encode(test_refs, convert_to_tensor=True)
    sem_score = float(cosine_similarity(emb_p.cpu().numpy(), emb_r.cpu().numpy()).diagonal().mean())

    wandb.log({
        "test_accuracy": test_acc,
        "test_exact_match": exact_match,
        "test_rouge1": rouge_res['rouge1'].mid.fmeasure,
        "test_rouge2": rouge_res['rouge2'].mid.fmeasure,
        "test_sem_score": sem_score,
        **{f"test_acc_{s}": c['correct']/c['total'] for s,c in test_counts.items()}
    })
    print(f"Test Acc: {test_acc:.4f}, EM: {exact_match:.4f}, "
          f"ROUGE1: {rouge_res['rouge1'].mid.fmeasure:.4f}, "
          f"ROUGE2: {rouge_res['rouge2'].mid.fmeasure:.4f}, "
          f"SemScore: {sem_score:.4f}")

    # (이하 추출형 QA 평가는 원래 코드 그대로 두시면 됩니다.)




    # 추출형 QA 평가
    ext = load_dataset('json', data_files={'test':'/home/wcjeong/my_project/TeleQuAD/extractive/v4/TeleQuAD_split/test.json'})['test']
    rouge_ext = evaluate.load('rouge')
    squad_ext = evaluate.load('squad')
    emb_ext = SentenceTransformer('all-MiniLM-L6-v2')
    ext_preds, ext_refs, sq_inputs = [], [], []
    for ex in ext:
        prompt_e = f"Context: {ex['context']}\n\nQuestion: {ex['question']}\nAnswer:"
        inp_e = tokenizer(prompt_e, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
        gen_e = model.generate(**inp_e, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)[0]
        p_e = tokenizer.decode(gen_e[inp_e.input_ids.size(1):], skip_special_tokens=True).strip()
        ext_preds.append(p_e)
        ext_refs.append(ex['answers'][0]['text'])
        sq_inputs.append({'id':str(ex['id']),'prediction_text':p_e,'answers':{'text':[a['text'] for a in ex['answers']],'answer_start':[a['answer_start'] for a in ex['answers']]}})
    rouge_res_ext = rouge_ext.compute(predictions=ext_preds, references=ext_refs)
    sq_preds_ext = [{'id':d['id'],'prediction_text':d['prediction_text']} for d in sq_inputs]
    sq_refs_ext = [{'id':d['id'],'answers':d['answers']} for d in sq_inputs]
    squad_res_ext = squad_ext.compute(predictions=sq_preds_ext, references=sq_refs_ext)
    emb_pe = emb_ext.encode(ext_preds, convert_to_tensor=True)
    emb_re = emb_ext.encode(ext_refs, convert_to_tensor=True)
    sem_e = float(cosine_similarity(emb_pe.cpu().numpy(), emb_re.cpu().numpy()).diagonal().mean())
    wandb.log({
        'ext_rouge1':rouge_res_ext['rouge1'],
        'ext_rouge2':rouge_res_ext['rouge2'],
        'ext_rougeL':rouge_res_ext['rougeL'],
        'ext_f1':squad_res_ext['f1'],
        'ext_exact_match':squad_res_ext['exact_match'],
        'ext_sem_score':sem_e
    })
    print(f"Extractive QA → ROUGE1: {rouge_res_ext['rouge1']:.4f}, ROUGE2: {rouge_res_ext['rouge2']:.4f}, ROUGEL: {rouge_res_ext['rougeL']:.4f}, F1: {squad_res_ext['f1']:.4f}, EM: {squad_res_ext['exact_match']:.4f}, SemScore: {sem_e:.4f}")

if __name__ == "__main__":
    main()