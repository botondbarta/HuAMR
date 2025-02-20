import logging

from dotmap import DotMap
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing_extensions import override

from huamr.s2s_models.base_model import S2SBaseModel

logger = logging.getLogger(__name__)


class MT5(S2SBaseModel):
    def __init__(self, config: DotMap):
        super().__init__(config)

        if self.config.load_model:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.config.load_model)
        else:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_checkpoint)
            for param in self._model.parameters():
                param.data = param.data.contiguous()

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_checkpoint, legacy=False, use_fast=True)

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
        if batch.has_key('amr_graph'):
            outputs = self._tokenizer(text_target=batch['amr_graph'],
                                      padding='max_length',
                                      max_length=self.config.max_output_length,
                                      truncation=True)

            outputs["input_ids"] = [
                [(l if l != self._tokenizer.pad_token_id else -100) for l in label] for label in outputs["input_ids"]
            ]
            inputs['labels'] = outputs['input_ids']
        return inputs
