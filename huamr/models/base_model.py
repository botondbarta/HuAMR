from abc import ABC

from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer

from huamr.utils import get_bnb_config


class BaseModel(ABC):
    def get_model_and_tokenizer(self, model_name, quantize, hf_token):
        model = self.get_model(model_name, quantize, hf_token)
        tokenizer = self.get_tokenizer(model_name, hf_token)

        self.set_special_tokens(model_name, tokenizer)

        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False

        model = prepare_model_for_kbit_training(model)
        return model, tokenizer

    def get_model(self, model_name, quantize, hf_token):
        return AutoModelForCausalLM.from_pretrained(model_name,
                                                    quantization_config=get_bnb_config(quantize),
                                                    device_map='auto',
                                                    token=hf_token)

    def get_tokenizer(self, model_name, hf_token):
        return AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side='left', token=hf_token)

    def set_special_tokens(self, model_name, tokenizer):
        pass
