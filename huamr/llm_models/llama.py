from typing import override

from huamr.llm_models.base_model import LLMBaseModel


class LlamaModel(LLMBaseModel):
    @override
    def set_special_tokens(self, model_name, tokenizer):
        if "llama-3." in model_name.lower():
            tokenizer.pad_token = '<|finetune_right_pad_id|>'
            tokenizer.bos_token = '<|begin_of_text|>'
            tokenizer.eos_token = '<|eot_id|>'