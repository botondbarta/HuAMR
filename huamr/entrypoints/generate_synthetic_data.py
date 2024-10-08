import os
from pathlib import Path

import click
import pandas as pd
from peft import PeftModel
from tqdm import tqdm

from huamr.utils.config_reader import get_config_from_yaml
from huamr.utils.model_factory import ModelFactory

HF_TOKEN = os.getenv('HF_TOKEN')


def batch_inference(wrapped_model, sentences, output_path, batch_size=32):
    all_outputs = []

    output_file = os.path.join(output_path, 'generated.csv')

    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i: i + batch_size]
        outputs = wrapped_model.inference(batch)
        all_outputs.extend(outputs)

        generated_amr_batch = [output.split('### AMR Graph')[-1].strip() for output in outputs]

        batch_df = pd.DataFrame({
            'sentence': batch,
            'generated_amr': generated_amr_batch
        })

        mode, header = ('w', True) if i == 0 else ('a', False)

        batch_df.to_csv(output_file, mode=mode, header=header, index=False)


@click.command()
@click.argument('config_path')
@click.argument('adapter_path')
@click.argument('input_file')
@click.argument('output_path')
@click.argument('batch_size', type=int, default=32)
def main(config_path, adapter_path, input_file, output_path, batch_size):
    config = get_config_from_yaml(config_path)
    adapter = Path(adapter_path)

    wrapped_model = ModelFactory.get_model(config, HF_TOKEN)
    wrapped_model.model = PeftModel.from_pretrained(wrapped_model.get_model(), adapter)
    wrapped_model.model.eval()

    df = pd.read_csv(input_file, header=0)
    sentences = df['sentence'].tolist()

    batch_inference(wrapped_model, sentences, output_path, batch_size)


if __name__ == '__main__':
    main()
