import logging

from dotmap import DotMap
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, EncoderDecoderModel, BertTokenizer, AutoConfig, AutoModel
from typing_extensions import override

from huamr.s2s_models.base_model import S2SBaseModel

logger = logging.getLogger(__name__)


class AMREncoderDecoderModel(S2SBaseModel):
    def __init__(self, config: DotMap):
        super().__init__(config)

        encoder_model = 'SZTAKI-HLT/hubert-base-cc'
        decoder_model = 'Qwen/Qwen2.5-Coder-1.5B'

        self.bert_tokenizer = BertTokenizer.from_pretrained(encoder_model)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model)

        encoder_config = AutoConfig.from_pretrained(encoder_model)
        decoder_config = AutoConfig.from_pretrained(decoder_model)

        # Modify decoder config to accept encoder hidden states
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        decoder_config.cross_attention_hidden_size = encoder_config.hidden_size

        encoder = AutoModel.from_pretrained(encoder_model)
        decoder = AutoModel.from_pretrained(
            decoder_model,
            config=decoder_config
        )

        # Initialize the encoder-decoder model
        self._model = EncoderDecoderModel(
            encoder=encoder,
            decoder=decoder
        )

        self._model.config.decoder_start_token_id = self.decoder_tokenizer.bos_token_id
        self._model.config.pad_token_id = self.decoder_tokenizer.pad_token_id
        self._model.config.eos_token_id = self.decoder_tokenizer.eos_token_id

        self._model.config.hidden_size = encoder_config.hidden_size
        self._model.config.decoder_hidden_size = decoder_config.hidden_size

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

        outputs = self.decoder_tokenizer(text_target=batch['amr_graph'],
                                       padding='max_length',
                                       max_length=self.config.max_output_length,
                                       truncation=True)

        outputs["input_ids"] = [
            [(l if l != self.decoder_tokenizer.pad_token_id else -100) for l in label] for label in outputs["input_ids"]
        ]
        inputs['labels'] = outputs['input_ids']
        return inputs
