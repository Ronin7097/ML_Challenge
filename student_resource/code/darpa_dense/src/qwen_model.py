"""Local Qwen yes/no pair scoring with a fixed business-matching instruction."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-Reranker-0.6B"
REVISION = "2925c98b11f00b3364acaeb0a669f498ac45bf54"
PREFIX = '<|im_start|>system\nUse the instruction to compare the query and document. Respond only with yes or no.<|im_end|>\n<|im_start|>user\n'
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
INSTRUCTION = "Decide whether both records identify the same business at the same location. Allow noisy spelling, abbreviations, transliteration and missing address parts. Distinguish neighboring businesses and different branches."


def load(base, adapter=None, training=False):
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=training)
    model.config.use_cache = False
    model.train(training)
    return tokenizer, model


def inputs(tokenizer, targets, sources, max_length=320):
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    budget = max_length-len(prefix)-len(suffix)
    if budget < 64:
        raise ValueError("Prompt leaves too few tokens for business records")
    bodies = [f"<Instruct>: {INSTRUCTION}\n<Query>: {t}\n<Document>: {s}" for t,s in zip(targets,sources)]
    encoded = tokenizer(bodies, add_special_tokens=False, truncation=True, max_length=budget,
                        return_attention_mask=False)["input_ids"]
    packed = {"input_ids": [prefix+x+suffix for x in encoded]}
    return tokenizer.pad(packed, padding=True, return_tensors="pt").to("cuda")


def logits(tokenizer, model, batch):
    yes, no = tokenizer.convert_tokens_to_ids("yes"), tokenizer.convert_tokens_to_ids("no")
    if yes == no or yes == tokenizer.unk_token_id or no == tokenizer.unk_token_id:
        raise ValueError("Tokenizer has no distinct yes/no labels")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1]
    return result[:, [no,yes]].float()
