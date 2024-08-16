from typing import override

from huamr.models.base_model import BaseModel


class MistralModel(BaseModel):
    @override
    def set_special_tokens(self, model_name, tokenizer):
        if 'mistral-7b-instruct-v0.3' in model_name.lower():
            tokenizer.pad_token = '[control_768]'