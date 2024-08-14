from peft import prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM

from huamr.utils import get_bnb_config


class ModelFactory:
    @staticmethod
    def get_model(model_name, quantize, HF_TOKEN):
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     quantization_config=get_bnb_config(quantize),
                                                     device_map='auto',
                                                     token=HF_TOKEN)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side='left', token=HF_TOKEN)

        if "llama-3.1" in model_name.lower():
            tokenizer.pad_token = '<|finetune_right_pad_id|>'
            tokenizer.eos_token = '<|eot_id|>'
            tokenizer.bos_token = '<|begin_of_text|>'
        if 'mistral-7b-instruct-v0.3' in model_name.lower():
            tokenizer.pad_token = '[control_768]'

        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        
        model = prepare_model_for_kbit_training(model)
        return model, tokenizer
