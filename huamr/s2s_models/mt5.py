import logging

from dotmap import DotMap
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing_extensions import override

from huamr.s2s_models.base_model import S2SBaseModel

logger = logging.getLogger(__name__)


class MT5(S2SBaseModel):
    def __init__(self, config: DotMap):
        super().__init__(config)

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_checkpoint, legacy=False, use_fast=True)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_checkpoint)

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
