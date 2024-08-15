import os
import shutil
from pathlib import Path

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
from huamr.utils.config_reader import get_config_from_yaml
from huamr.utils.langtype import LangType
from huamr.utils.model_factory import ModelFactory

HF_TOKEN = os.getenv('HF_TOKEN')


def load_dataset(config, eos_token):
    dataset = AMR3Dataset(config.data_path, LangType[config.language], config.remove_wiki)
    train, validation, test = dataset.get_split()
    dataset = DatasetDict({
        'train': Dataset.from_pandas(pd.DataFrame(train)),
        'validation': Dataset.from_pandas(pd.DataFrame(validation)),
        'test': Dataset.from_pandas(pd.DataFrame(test)),
    })

    prompt = """### Instruction
Provide the AMR graph for the following sentence. Ensure that the graph captures the main concepts, the relationships between them, and any additional information that is important for understanding the meaning of the sentence. Use standard AMR notation, including concepts, roles, and relationships.

### Sentence
{}

### AMR Graph
{}"""

    def formatting_prompts_func(examples):
        sentences = examples["sentence"]
        amr_graphs = examples["amr_graph"]
        texts = []
        for sentence, amr_graph in zip(sentences, amr_graphs):
            text = prompt.format(sentence, amr_graph) + eos_token
            texts.append(text)
        return {"text": texts, }

    dataset = dataset.map(formatting_prompts_func, batched=True, )

    return dataset


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

    model, tokenizer = ModelFactory.get_model(config.model_name, config.quantize, HF_TOKEN)

    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )

    dataset = load_dataset(config, tokenizer.eos_token)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        tokenizer=tokenizer,
        args=get_training_arg(config),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.patience)]
    )
    trainer.train()

    shutil.copy2(config_path, Path(config.output_dir)/Path(config_path).name)


if __name__ == "__main__":
    main()
