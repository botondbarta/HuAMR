import os

import amrlib
import click
import torch
from tqdm import tqdm


def batch_inference(stog_model, europarl_data, output_path, batch_size=32):
    output_file = os.path.join(output_path, 'europarl_generated.txt')

    for i in tqdm(range(0, len(europarl_data), batch_size)):
        en_sentences = [row[0] for row in europarl_data[i: i + batch_size]]
        hu_sentences = [row[1] for row in europarl_data[i: i + batch_size]]

        outputs = stog_model.parse_sents(en_sentences)
        outputs = [f'# ::snt_hu {hu_sentence}\n' + output for output, hu_sentence in zip(outputs, hu_sentences)]

        mode = 'w' if i == 0 else 'a'

        with open(output_file, mode) as f:
            f.write('\n\n'.join(outputs))
            f.write('\n\n')


def load_europarl_data(europarl_folder):
    en_file = f'{europarl_folder}/europarl-v7.hu-en.en'
    hu_file = f'{europarl_folder}/europarl-v7.hu-en.hu'

    with open(en_file, 'r') as f:
        en_lines = [line.strip() for line in f.readlines()]

    with open(hu_file, 'r') as f:
        hu_lines = [line.strip() for line in f.readlines()]

    assert len(en_lines) == len(hu_lines)

    return [(en_line, hu_line) for en_line, hu_line in zip(en_lines, hu_lines)]


@click.command()
@click.argument('stog_model_dir')
@click.argument('europarl_folder')
@click.argument('output_path')
@click.argument('batch_size', type=int, default=32)
def main(stog_model_dir, europarl_folder, output_path, batch_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = load_europarl_data(europarl_folder)

    stog_model = amrlib.load_stog_model(model_dir=stog_model_dir, device=device)

    batch_inference(stog_model, data, output_path, batch_size)


if __name__ == '__main__':
    main()
