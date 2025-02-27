from abc import ABC

from dotmap import DotMap
from peft import prepare_model_for_kbit_training
from smatchpp import Smatchpp, solvers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from huamr.utils import get_bnb_config
from huamr.utils.constants import SYSTEM_PROMPT


class LLMBaseModel(ABC):
    def __init__(self, config: DotMap, hf_token, do_train: bool = False):
        self.config = config
        self.hf_token = hf_token

        self.model = self.load_model(config.model_name, config.quantize, hf_token)
        self.tokenizer = self.load_tokenizer(config.model_name, config.hf_token)

        self.set_special_tokens(config.model_name, self.tokenizer)

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.use_cache = False

        if config.quantize and do_train:
            self.model = prepare_model_for_kbit_training(self.model)

        ilp = solvers.ILP()
        self.measure = Smatchpp(alignmentsolver=ilp)

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def load_model(self, model_name, quantize, hf_token):
        return AutoModelForCausalLM.from_pretrained(model_name,
                                                    quantization_config=get_bnb_config(quantize) if quantize else None,
                                                    device_map='auto',
                                                    token=hf_token)

    def load_tokenizer(self, model_name, hf_token):
        return AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side='left', token=hf_token)

    def inference(self, sentences: list[str]) -> list[str]:
        prompts = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sentence},
            ] for sentence in sentences
        ]
        inputs = self.tokenizer.apply_chat_template(prompts, padding=True, return_tensors="pt").to("cuda")

        generation_config = GenerationConfig(
            num_beams=5,
            return_dict_in_generate=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=self.config.generate_max_length,
        )
        outputs = self.model.generate(**inputs, generation_config=generation_config)
        return self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=True)

    def set_special_tokens(self, model_name, tokenizer):
        pass
