import os
import shutil
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from datasets import DatasetDict, Dataset
from peft import LoraConfig
from transformers import (
    TrainingArguments,
    IntervalStrategy,
    EarlyStoppingCallback,
)
from trl import SFTTrainer

from huamr.data.amr3 import AMR3Dataset
from huamr.utils.amr_helper import filter_valid_amrs
from huamr.utils.config_reader import get_config_from_yaml
from huamr.utils.constants import sentence_to_amr_prompt
from huamr.utils.langtype import LangType
from huamr.utils.model_factory import ModelFactory

HF_TOKEN = os.getenv('HF_TOKEN')


def load_synthetic_data(file) -> Optional[pd.DataFrame]:
    if file:
        df = pd.read_csv(file)
        df = df.rename(columns={'generated_amr': 'amr_graph'})
        df = filter_valid_amrs(df, 'amr_graph')

        return df

    return None


def load_dataset(config, eos_token):
    dataset = AMR3Dataset(config.data_path, config.remove_wiki)
    train, validation, _ = dataset.get_split(LangType[config.train_language], LangType[config.dev_language])

    synthetic_data = load_synthetic_data(config.synthetic_data)

    dataset = DatasetDict({
        'train': Dataset.from_pandas(pd.concat([pd.DataFrame(train), synthetic_data])),
        'validation': Dataset.from_pandas(pd.DataFrame(validation)),
    })

    def format_sentence_to_amr(examples):
        sentences = examples["sentence"]
        amr_graphs = examples["amr_graph"]
        texts = []
        for sentence, amr_graph in zip(sentences, amr_graphs):
            text_sentence_to_amr = sentence_to_amr_prompt.format(sentence, amr_graph) + eos_token
            texts.append(text_sentence_to_amr)
        return {"text": texts, }

    dataset_s2a = dataset.map(format_sentence_to_amr, batched=True, )

    return DatasetDict({
        'train': dataset_s2a['train'],
        'validation': dataset_s2a['validation'],
    })


def get_training_arg(config):
    return TrainingArguments(
        output_dir=config.output_dir,
        do_eval=True,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,

        log_level='debug',
        optim='paged_adamw_32bit',

        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,

        evaluation_strategy=IntervalStrategy.STEPS,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        logging_steps=config.logging_steps,

        max_grad_norm=config.max_grad_norm,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        group_by_length=config.group_by_length,

        lr_scheduler_type='constant',
        bf16=True,
        report_to=None,
    )


@click.command()
@click.argument('config_path')
def main(config_path):
    config = get_config_from_yaml(config_path)

    wrapped_model = ModelFactory.get_model(config, HF_TOKEN, do_train=True)

    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )

    dataset = load_dataset(config, wrapped_model.get_tokenizer().eos_token)

    trainer = SFTTrainer(
        model=wrapped_model.get_model(),
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        tokenizer=wrapped_model.get_tokenizer(),
        args=get_training_arg(config),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.patience)]
    )
    trainer.train()

    shutil.copy2(config_path, Path(config.output_dir) / Path(config_path).name)


if __name__ == "__main__":
    main()
