import os
from pathlib import Path

import click
import pandas as pd
from datasets import DatasetDict, Dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import Seq2SeqTrainer

from huamr.data.amr3 import AMR3Dataset
from huamr.entrypoints.train_s2s import get_training_arg
from huamr.s2s_models.base_model import S2SBaseModel
from huamr.utils.config_reader import get_config_from_yaml
from huamr.utils.langtype import LangType
from huamr.utils.model_factory import ModelFactory, S2SModelFactory

HF_TOKEN = os.getenv('HF_TOKEN')



def load_dataset(config, model: S2SBaseModel):
    dataset = AMR3Dataset(config.data_path, config.remove_wiki)
    _, _, test = dataset.get_split(test_lang=LangType[config.test_language])
    dataset = DatasetDict({'test': Dataset.from_pandas(pd.DataFrame(test))})

    dataset = dataset.map(model.process_data_to_model_inputs, batched=True, )

    return dataset


@click.command()
@click.argument('config_path')
@click.argument('output_path')
def main(config_path, output_path):
    config = get_config_from_yaml(config_path)

    model: S2SBaseModel = S2SModelFactory.get_model(config)
    model.get_model().eval()

    test_set = load_dataset(config, model)

    trainer = Seq2SeqTrainer(
        model=model.get_model(),
        tokenizer=model.get_tokenizer(),
        args=get_training_arg(config),
    )

    test_outputs = trainer.predict(
        test_dataset=test_set['test'],
        max_length=1024,
    )

    predictions = test_outputs.predictions
    predictions[predictions == -100] = model.get_tokenizer().pad_token_id
    test_preds = model.get_tokenizer().batch_decode(
        predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    test_set = pd.DataFrame(test_set['test'])
    test_set['generated_amr'] = list(map(str.strip, test_preds))
    test_set.to_csv(os.path.join(output_path, 'generated.csv'), header=True, index=False)


if __name__ == '__main__':
    main()
