from typing import override

from huamr.llm_models.base_model import LLMBaseModel


class NvidiaModel(LLMBaseModel):
    @override
    def set_special_tokens(self, model_name, tokenizer):
        if "nemotron-mini-4b-instruct" in model_name.lower():
            tokenizer.pad_token = '<pad>'