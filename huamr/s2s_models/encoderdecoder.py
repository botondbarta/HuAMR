import logging

from dotmap import DotMap
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, EncoderDecoderModel, BertTokenizer
from typing_extensions import override

from huamr.s2s_models.base_model import S2SBaseModel

logger = logging.getLogger(__name__)


class AMREncoderDecoderModel(S2SBaseModel):
    def __init__(self, config: DotMap):
        super().__init__(config)


        self.bert_tokenizer = BertTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc')
        self.coder_tokenizer = AutoTokenizer.from_pretrained("qwen2.5-coder")


        self._model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            'SZTAKI-HLT/hubert-base-cc',
            'Qwen/Qwen2.5-Coder-1.5B'
        )

    @override
    def get_model(self):
        return self._model

    @override
    def get_tokenizer(self):
        raise NotImplementedError

    @override
    def process_data_to_model_inputs(self, batch):
        inputs = self.bert_tokenizer(batch['sentence'],
                                 padding='max_length',
                                 max_length=self.config.max_input_length,
                                 truncation=True)

        outputs = self.coder_tokenizer(text_target=batch['amr_graph'],
                                  padding='max_length',
                                  max_length=self.config.max_output_length,
                                  truncation=True)

        outputs["input_ids"] = [
            [(l if l != self.coder_tokenizer.pad_token_id else -100) for l in label] for label in outputs["input_ids"]
        ]
        inputs['labels'] = outputs['input_ids']
        return inputs
