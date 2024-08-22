import logging
from abc import ABC, abstractmethod

from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from huamr.utils.config_reader import get_config_from_yaml

logger = logging.getLogger(__name__)


class S2SBaseModel(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def get_model(self):
        raise NotImplementedError

    @abstractmethod
    def get_tokenizer(self):
        raise NotImplementedError

    @abstractmethod
    def process_data_to_model_inputs(self, batch):
        raise NotImplementedError
