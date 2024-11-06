from dotmap import DotMap

from huamr.llm_models.base_model import LLMBaseModel
from huamr.llm_models.gemma import GemmaModel
from huamr.llm_models.llama import LlamaModel
from huamr.llm_models.mistral import MistralModel
from huamr.llm_models.nvidia import NvidiaModel
from huamr.llm_models.phi import PhiModel
from huamr.s2s_models.aya101 import Aya101
from huamr.s2s_models.base_model import S2SBaseModel
from huamr.s2s_models.mt5 import MT5
from huamr.s2s_models.mbart50 import mBART50


class ModelFactory:
    @staticmethod
    def get_model(config: DotMap, hf_token, do_train: bool = False) -> LLMBaseModel:
        model_map = {
            'llama': LlamaModel,
            'mistral': MistralModel,
            'gemma': GemmaModel,
            'phi': PhiModel,
            'nvidia': NvidiaModel,
        }

        for key, model_class in model_map.items():
            if key in config.model_name.lower():
                return model_class(config, hf_token, do_train)

        return LLMBaseModel(config, hf_token, do_train)


class S2SModelFactory:
    @staticmethod
    def get_model(config: DotMap) -> S2SBaseModel:
        model_map = {
            'mt5': MT5,
            'aya': Aya101,
            'mbart': mBART50,
        }

        for key, model_class in model_map.items():
            if key in config.model_checkpoint.lower():
                return model_class(config)

        raise Exception('Model not found')
