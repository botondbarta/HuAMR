import logging

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, EarlyStoppingCallback
from typing_extensions import override

from huamr.s2s_models.base_model import S2SBaseModel

logger = logging.getLogger(__name__)


class MT5(S2SBaseModel):
    def __init__(self, config_path):
        super().__init__(config_path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_checkpoint)

    @override
    def get_model(self):
        return self.model

    @override
    def get_tokenizer(self):
        return self.tokenizer

    @override
    def process_data_to_model_inputs(self, batch):
        inputs = self.tokenizer(batch['sentence'],
                                padding='max_length',
                                max_length=self.config.max_input_length,
                                truncation=True)

        outputs = self.tokenizer(text_target=batch['amr_graph'],
                                 padding='max_length',
                                 max_length=self.config.max_output_length,
                                 truncation=True)

        outputs["input_ids"] = [
            [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in outputs["input_ids"]
        ]
        inputs['labels'] = outputs['input_ids']
        return inputs
