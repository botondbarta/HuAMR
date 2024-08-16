from huamr.models.base_model import BaseModel
from huamr.models.gemma import GemmaModel
from huamr.models.llama import LlamaModel
from huamr.models.mistral import MistralModel


class ModelFactory:
    @staticmethod
    def get_model(model_name, quantize, hf_token):
        model_map = {
            'llama': LlamaModel,
            'mistral': MistralModel,
            'gemma': GemmaModel
        }

        for key, model_class in model_map.items():
            if key in model_name.lower():
                return model_class().get_model_and_tokenizer(model_name, quantize, hf_token)

        return BaseModel().get_model_and_tokenizer(model_name, quantize, hf_token)
