"""
ARIS V10.1 — Merge and Quantize Script
"""
from unsloth import FastLanguageModel
from peft import PeftModel

MODEL_NAME  = "unsloth/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "provenance/adapter"
OUTPUT_DIR  = "ARIS-V10.1"
QUANT       = "q4_k_m"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME, max_seq_length=2048, load_in_4bit=True, dtype=None,
)
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.save_pretrained_gguf(OUTPUT_DIR, tokenizer, quantization_method=QUANT)
print(f"GGUF written to {OUTPUT_DIR}_gguf/")
