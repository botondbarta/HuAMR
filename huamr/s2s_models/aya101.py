import logging

import peft
from dotmap import DotMap
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing_extensions import override

from huamr.s2s_models.base_model import S2SBaseModel
from huamr.utils import get_bnb_config

logger = logging.getLogger(__name__)


class Aya101(S2SBaseModel):
    def __init__(self, config: DotMap):
        super().__init__(config)

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_checkpoint, legacy=False, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_checkpoint,
                                                      quantization_config=get_bnb_config(config.quantize),
                                                      device_map='auto',
                                                      )
        model = peft.prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q", "v"]
        )

        self._model = get_peft_model(model, peft_config)

    @override
    def get_model(self):
        return self._model

    @override
    def get_tokenizer(self):
        return self._tokenizer

    @override
    def process_data_to_model_inputs(self, batch):
        inputs = self._tokenizer(batch['sentence'],
                                 padding='max_length',
                                 max_length=self.config.max_input_length,
                                 truncation=True)

        outputs = self._tokenizer(text_target=batch['amr_graph'],
                                  padding='max_length',
                                  max_length=self.config.max_output_length,
                                  truncation=True)

        outputs["input_ids"] = [
            [(l if l != self._tokenizer.pad_token_id else -100) for l in label] for label in outputs["input_ids"]
        ]
        inputs['labels'] = outputs['input_ids']
        return inputs
