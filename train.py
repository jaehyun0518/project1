
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import bitsandbytes as bnb
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer

api_token="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"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
new_model = "./model/llama3-8b-instruct-finetune"

# QLoRA config
torch_dtype = torch.float16
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=False,
)

# Step 1: 토크나이저 및 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map={"": 0}
    # device_map="auto"
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# 패딩 토큰 설정
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"    # 옵션

# Step 2: 데이터셋 로드 및 준비
dataset = load_dataset('json', data_files={'train': './train_data-textbook.json', 'test': './test_data.json'})
print(dataset['train'][0])

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

tokenized_datasets = dataset.map(preprocess_function, batched=True)


peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# Model Lora

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,
#     per_device_train_batch_size=4,
#     save_steps=10_000,
#     save_total_limit=2,
#     logging_dir='./logs',
#     logging_steps=500,
# )

# Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     peft_config=peft_params,
#     train_dataset=tokenized_datasets['train'],
#     eval_dataset=tokenized_datasets['test'],
# )
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)

# Step 4: 학습 실행
trainer.train()
trainer.save_model(new_model)