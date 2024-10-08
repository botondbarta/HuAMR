import logging
from abc import ABC, abstractmethod

from dotmap import DotMap

from huamr.utils.smatch import calculate_smatch

logger = logging.getLogger(__name__)


class S2SBaseModel(ABC):
    def __init__(self, config: DotMap):
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

    def compute_metrics(self, pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        labels_ids[labels_ids == -100] = self.get_tokenizer().pad_token_id
        pred_ids[pred_ids == -100] = self.get_tokenizer().pad_token_id

        pred_graphs = self.get_tokenizer().batch_decode(pred_ids, skip_special_tokens=True)
        ref_graphs = self.get_tokenizer().batch_decode(labels_ids, skip_special_tokens=True)

        smatch_score = calculate_smatch(ref_graphs, pred_graphs)

        return {**smatch_score}



