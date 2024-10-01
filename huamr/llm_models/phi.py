from typing import override

from transformers import AutoModelForCausalLM

from huamr.llm_models.base_model import LLMBaseModel
from huamr.utils import get_bnb_config


class PhiModel(LLMBaseModel):
    @override
    def load_model(self, model_name, quantize, hf_token):
        return AutoModelForCausalLM.from_pretrained(model_name,
                                                    quantization_config=get_bnb_config(quantize),
                                                    device_map='auto',
                                                    trust_remote_code=True,
                                                    attn_implementation="flash_attention_2",
                                                    token=hf_token)
