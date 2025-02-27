import os
from pathlib import Path

import click
import pandas as pd
from dotmap import DotMap
from peft import PeftModel
from tqdm import tqdm

from huamr.data.amr3 import AMR3Dataset
from huamr.utils.config_reader import get_config_from_yaml
from huamr.utils.langtype import LangType
from huamr.utils.model_factory import ModelFactory

HF_TOKEN = os.getenv('HF_TOKEN')


def batch_inference(wrapped_model, sentences, batch_size=32):
    all_outputs = []
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i: i + batch_size]
        outputs = wrapped_model.inference(batch)
        all_outputs.extend(outputs)
    return all_outputs


def load_dataset(test_dataset, config: DotMap) -> pd.DataFrame:
    if test_dataset == 'amr':
        dataset = AMR3Dataset(config.data_path)
        _, _, test_set = dataset.get_split(test_lang=LangType[config.test_language])
        return pd.DataFrame(test_set)

    elif test_dataset == 'huamr':
        df = pd.read_csv(config.synthetic_data)
        df = df.rename(columns={'generated_amr': 'amr_graph'})
        df = df.iloc[40000:]  # last 3-4k examples are for testing
        return df
    else:
        raise Exception('Not a vaild dataset')


@click.command()
@click.argument('config_path')
@click.argument('output_path')
@click.argument('batch_size', type=int, default=32)
@click.option('--adapter_path', default=None)
@click.option('-t', '--test_dataset', default='amr')
def main(config_path, output_path, batch_size, adapter_path, test_dataset):
    config = get_config_from_yaml(config_path)

    wrapped_model = ModelFactory.get_model(config, HF_TOKEN)
    if adapter_path:
        adapter = Path(adapter_path)
        wrapped_model.model = PeftModel.from_pretrained(wrapped_model.get_model(), adapter)
    wrapped_model.model.eval()

    test_set = load_dataset(test_dataset, config)

    sentences = test_set['sentence'].tolist()
    generated_outputs = batch_inference(wrapped_model, sentences, batch_size)
    test_set['generated_amr'] = [output.split('assistant\n\n')[-1].strip() for output in generated_outputs]

    test_set.to_csv(os.path.join(output_path, 'generated.csv'), header=True, index=False)


if __name__ == '__main__':
    main()
