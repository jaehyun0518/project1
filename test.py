import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import bitsandbytes as bnb
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel
from trl import SFTTrainer

api_token="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"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
new_model = "./model/llama3-8b-instruct-finetune"


tokenizer = AutoTokenizer.from_pretrained(model_id)

# 패딩 토큰 설정
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"    # 옵션

# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     low_cpu_mem_usage=True,
#     return_dict=True,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )

torch_dtype = torch.float16
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=False,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map={"": 0},
    torch_dtype=torch.float16
    # device_map="auto"
)

# model, tokenizer = setup_chat_format(model, tokenizer)

# Merge adapter with base model
model = PeftModel.from_pretrained(model, new_model)
model = model.merge_and_unload()


# tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=api_token)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     use_auth_token=api_token
# )

messages = [
    # [
    #     {"role": "system", "content": "You are the tutor who always has to answer science questions."},
    #     {"role": "user", "content": "Tell me more about the solar energy"},
    # ],
    # [
    #     {"role": "system", "content": "You are the tutor who always has to answer science questions."},
    #     {"role": "user", "content": "Let's take a deep breath and start. Tell me more about the sun"},
    # ],
    [
        {"role": "system", "content": "You are the tutor who always has to answer science questions."},
        {"role": "user", "content": "Tell me more about the sun"},
    ]
]

# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"},
# ]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    padding=True,
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

print("=======================================================")
for i in range(len(outputs)):
    response = outputs[i][input_ids.shape[-1]:]
    print(tokenizer.decode(response, skip_special_tokens=True))
    print("=======================================================\n")