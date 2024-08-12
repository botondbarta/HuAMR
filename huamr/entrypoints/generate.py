import os
from pathlib import Path

import click
import pandas as pd
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig

from huamr.data.amr3 import AMR3Dataset
from huamr.utils.config_reader import get_config_from_yaml
from huamr.utils.langtype import LangType
from huamr.utils.model_factory import ModelFactory

tqdm.pandas()
HF_TOKEN = os.getenv('HF_TOKEN')


def inference(model, tokenizer, sentence, max_new_tokens=1024, temperature=0.1, num_beams=5):
    prompt = f"""### Instruction
Provide the AMR graph for the following sentence. Ensure that the graph captures the main concepts, the relationships between them, and any additional information that is important for understanding the meaning of the sentence. Use standard AMR notation, including concepts, roles, and relationships.

### Sentence
{sentence}

### AMR Graph
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    generation_config = GenerationConfig(
        do_sample=True,
        num_beams=num_beams,
        temperature=temperature,
        return_dict_in_generate=True,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )
    outputs = model.generate(**inputs, generation_config=generation_config)
    return tokenizer.decode(outputs.sequences[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)


def load_dataset(dataset_path):
    dataset = AMR3Dataset(dataset_path, LangType['HU'])
    _, _, test_set = dataset.get_split()
    return pd.DataFrame(test_set)


@click.command()
@click.argument('config_path')
@click.argument('adapter_path')
@click.argument('output_path')
def main(config_path, adapter_path, output_path):
    config = get_config_from_yaml(config_path)
    adapter = Path(adapter_path)

    model, tokenizer = ModelFactory.get_model(config.model_name, config.quantize, HF_TOKEN)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    test_set = load_dataset(config.data_path)
    test_set['generated_amr'] = test_set['sentence'].progress_apply(lambda x: inference(model, tokenizer, x))

    test_set.to_csv(os.path.join(output_path, 'generated.csv'), header=True, index=False)


if __name__ == '__main__':
    main()
