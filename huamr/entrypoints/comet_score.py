import subprocess
from pathlib import Path

import click
import pandas as pd
from comet import download_model, load_from_checkpoint

# subprocess.run('huggingface-cli login', shell=True)


def comet_score(data):
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)

    model_output = model.predict(data, batch_size=16, gpus=0)

    return model_output


def read_csvs(input_folder: str):
    input_folder = Path(input_folder)

    dfs = []
    for file in input_folder.iterdir():
        df = pd.read_csv(file)
        dfs.append(df)

    return pd.concat(dfs)


@click.command()
@click.argument('input_folder')
def main(input_folder):
    df = read_csvs(input_folder)

    data = [{"src": row["sentence"], "mt": row["hu_sentence"]} for _, row in df.iterrows()]

    score = comet_score(data)
    print(score)


if __name__ == '__main__':
    main()
