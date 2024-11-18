import os

import click
import pandas as pd
from datasets import DatasetDict, Dataset
from transformers import Seq2SeqTrainer

from huamr.data.amr3 import AMR3Dataset
from huamr.entrypoints.train_s2s import get_training_arg
from huamr.s2s_models.base_model import S2SBaseModel
from huamr.utils.config_reader import get_config_from_yaml
from huamr.utils.langtype import LangType
from huamr.utils.model_factory import S2SModelFactory

HF_TOKEN = os.getenv('HF_TOKEN')


def load_dataset(test_dataset, config):
    if test_dataset == 'amr':
        dataset = AMR3Dataset(config.data_path, config.remove_wiki)
        _, _, test_set = dataset.get_split(test_lang=LangType['HU'])
        return DatasetDict({'test': Dataset.from_pandas(pd.DataFrame(test_set))})

    elif test_dataset == 'huamr':
        df = pd.read_csv(config.synthetic_data)
        df = df.rename(columns={'generated_amr': 'amr_graph'})
        df = df.iloc[40000:]  # last 3-4k examples are for testing
        return DatasetDict({'test': Dataset.from_pandas(pd.DataFrame(df))})
    else:
        raise Exception('Not a vaild dataset')


@click.command()
@click.argument('config_path')
@click.argument('output_path')
@click.option('-t', '--test_dataset', default='amr')
def main(config_path, output_path, test_dataset):
    config = get_config_from_yaml(config_path)

    model: S2SBaseModel = S2SModelFactory.get_model(config)
    model.get_model().eval()

    test_set = load_dataset(test_dataset, config)
    test_set = test_set.map(model.process_data_to_model_inputs, batched=True, )

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
