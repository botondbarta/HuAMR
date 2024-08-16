from typing import override

from huamr.models.base_model import BaseModel


class LlamaModel(BaseModel):
    @override
    def set_special_tokens(self, model_name, tokenizer):
        if "llama-3.1" in model_name.lower():
            tokenizer.pad_token = '<|finetune_right_pad_id|>'
            tokenizer.eos_token = '<|eot_id|>'
            tokenizer.bos_token = '<|begin_of_text|>'