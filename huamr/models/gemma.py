from typing import override

from transformers import AutoModelForCausalLM

from huamr.models.base_model import BaseModel
from huamr.utils import get_bnb_config


class GemmaModel(BaseModel):
    @override
    def get_model(self, model_name, quantize, hf_token):
        return AutoModelForCausalLM.from_pretrained(model_name,
                                                    quantization_config=get_bnb_config(quantize),
                                                    device_map='auto',
                                                    attn_implementation='eager',
                                                    token=hf_token)