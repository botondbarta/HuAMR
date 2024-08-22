from huamr.llm_models.base_model import LLMBaseModel
from huamr.llm_models.gemma import GemmaModel
from huamr.llm_models.llama import LlamaModel
from huamr.llm_models.mistral import MistralModel
from huamr.llm_models.phi import PhiModel
from huamr.s2s_models.base_model import S2SBaseModel
from huamr.s2s_models.mt5 import MT5


class ModelFactory:
    @staticmethod
    def get_model(model_name, quantize, hf_token) -> LLMBaseModel:
        model_map = {
            'llama': LlamaModel,
            'mistral': MistralModel,
            'gemma': GemmaModel,
            'phi': PhiModel,
        }

        for key, model_class in model_map.items():
            if key in model_name.lower():
                return model_class().get_model_and_tokenizer(model_name, quantize, hf_token)

        return LLMBaseModel().get_model_and_tokenizer(model_name, quantize, hf_token)


class S2SModelFactory:
    @staticmethod
    def get_model(config) -> S2SBaseModel:
        model_map = {
            'mt5': MT5,
        }

        for key, model_class in model_map.items():
            if key in config.model_checkpoint.lower():
                return model_class(config)

        raise Exception('Model not found')
