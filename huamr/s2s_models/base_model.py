import logging
from abc import ABC, abstractmethod

from dotmap import DotMap

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
