import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from transformers.integrations import WandbCallback
from peft import PromptEncoderConfig, get_peft_model
import wandb

def main():

    #1: wandb 초기화
    wandb.init(project='teleqna-eval', name='qwen3-p-tuning-multiple_choice')

    #2: 데이터 로드
    ds = load_dataset('netop/TeleQnA', split='test')
    split1 = ds.train_test_split(test_size=0.3, seed=42)
    train_ds, rem_ds = split1['train'], split1['test']
    split2 = rem_ds.train_test_split(test_size=0.1, seed=42)
    val_ds, test_ds = split2['train'], split2['test']

    #3: 토크나이저 & 모델 로드
    model_name = "Qwen/Qwen3-1.7B-Base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto"
    )

    #4: P-Tuning (PromptEncoder) 설정 — PromptEncoderConfig 직접 전달
    p_tuning_config = PromptEncoderConfig(
        peft_type="P_TUNING",
        task_type="CAUSAL_LM",
        base_model_name_or_path=model_name,
        num_virtual_tokens=20,
        token_dim=model.config.hidden_size,
        num_transformer_submodules=1,
        num_attention_heads=model.config.num_attention_heads,
        num_layers=model.config.num_hidden_layers,
        encoder_reparameterization_type="MLP",
        encoder_hidden_size=model.config.hidden_size,
    )
    model = get_peft_model(model, p_tuning_config)

    #5: GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    #6: 전처리 함수 정의
    def preprocess_lm(examples):
        letters = ['1','2','3','4','5']
        prompts, labels = [], []
        for q, opts, a in zip(examples['question'], examples['choices'], examples['answer']):
            choice_str = "  ".join(f"[{letters[i]}] {opt}" for i, opt in enumerate(opts))
            prompts.append(f"Q: {q}\nChoices: {choice_str}\nAnswer:")
            labels.append(letters[a])
        return tokenizer(prompts, labels, truncation=True, padding="max_length", max_length=128)

    #6-1: datacollator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    #7: TrainingArguments 정의
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-4,
        fp16=True,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=500,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=500,
        report_to=["wandb"]
    )

    #8: Trainer 정의
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds.map(preprocess_lm, batched=True, remove_columns=train_ds.column_names),
        eval_dataset=val_ds.map(preprocess_lm, batched=True, remove_columns=val_ds.column_names),
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[WandbCallback]
    )

    #9: 학습 실행
    trainer.train()

    #10: validation loss 기록
    eval_results = trainer.evaluate()
    val_loss = eval_results["eval_loss"]
    print(f"Final Validation Loss: {val_loss:.4f}")
    wandb.log({"final_validation_loss": val_loss})

    #11: multiple‐choice 스코어링 함수
    def score_general(example):
        letters = ['1','2','3','4','5']
        prompt = (
            f"Q: {example['question']}\nChoices: " +
            "  ".join(f"[{letters[i]}] {opt}" for i, opt in enumerate(example['choices'])) +
            "\nAnswer:"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        last_logit = logits[0, -1]
        scores = [
            last_logit[tokenizer.convert_tokens_to_ids(letters[i])].item()
            for i in range(len(example['choices']))
        ]
        return int(np.argmax(scores))

    #12: Validation accuracy 계산
    correct = total = 0
    for ex in val_ds:
        if score_general(ex) == ex["answer"]:
            correct += 1
        total += 1
    val_acc = correct / total
    print(f"Validation Accuracy: {val_acc:.4f}")
    wandb.log({"validation_accuracy": val_acc})

    #13: Test accuracy 계산
    correct = total = 0
    for ex in test_ds:
        if score_general(ex) == ex["answer"]:
            correct += 1
        total += 1
    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")
    wandb.log({"test_accuracy": test_acc})

if __name__ == "__main__":
    main()
