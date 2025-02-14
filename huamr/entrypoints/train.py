import os
import shutil
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from datasets import DatasetDict, Dataset
from peft import LoraConfig
from transformers import IntervalStrategy, EarlyStoppingCallback
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from trl.trainer.sft_config import SFTConfig

from huamr.data.amr3 import AMR3Dataset
from huamr.utils.amr_validator import AMRValidator
from huamr.utils.config_reader import get_config_from_yaml
from huamr.utils.constants import shorter_prompt, sentence_to_amr_prompt
from huamr.utils.langtype import LangType
from huamr.utils.model_factory import ModelFactory

HF_TOKEN = os.getenv('HF_TOKEN')


def load_synthetic_data(file, synthetic_data_amount, frame_arg_descr) -> Optional[pd.DataFrame]:
    if file:
        df = pd.read_csv(file)
        df = df.rename(columns={'generated_amr': 'amr_graph'})
        df = df.iloc[:40000] # last 3-4k examples are for testing

        if frame_arg_descr:
            amr_validator = AMRValidator(frame_arg_descr)
            df = df[df['amr_graph'].apply(amr_validator.validate)]

        df = df.iloc[:synthetic_data_amount]

        return df

    return None


def load_dataset(config):
    train_df = pd.DataFrame()

    dataset = AMR3Dataset(config.data_path, config.remove_wiki)
    train, validation, _ = dataset.get_split(LangType[config.train_language], LangType[config.dev_language])

    if config.load_amr3:
        train_df = pd.DataFrame(train)
        train_df = train_df.sample(frac=1).iloc[:config.gold_data_amount]

    synthetic_data = load_synthetic_data(config.synthetic_data, config.synthetic_data_amount, config.frame_arg_descr)

    concatenated = pd.concat([train_df, synthetic_data])
    concatenated = concatenated.sample(frac=1).reset_index(drop=True)

    dataset = DatasetDict({
        'train': Dataset.from_pandas(concatenated),
        'validation': Dataset.from_pandas(pd.DataFrame(validation)),
    })

    return dataset


def format_dataset(dataset: DatasetDict, eos_token):
    def format_sentence_to_amr(examples):
        sentences = examples["sentence"]
        amr_graphs = examples["amr_graph"]
        texts = []
        for sentence, amr_graph in zip(sentences, amr_graphs):
            text_sentence_to_amr = sentence_to_amr_prompt.format(sentence, amr_graph) + eos_token
            texts.append(text_sentence_to_amr)
        return {"text": texts, }

    return dataset.map(format_sentence_to_amr, batched=True, )


def get_peft_config(config):
    return LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias='none',
        use_rslora=config.use_rslora,
        task_type='CAUSAL_LM',
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )


def get_training_arg(config):
    return SFTConfig(
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
        warmup_steps=config.warmup_steps,
        group_by_length=config.group_by_length,

        # metric_for_best_model='eval_smatch_f1',
        # greater_is_better=True,

        lr_scheduler_type='linear',
        bf16=True,
        report_to=None,
    )


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


@click.command()
@click.argument('config_path')
def main(config_path):
    config = get_config_from_yaml(config_path)

    wrapped_model = ModelFactory.get_model(config, HF_TOKEN, do_train=True)

    dataset = load_dataset(config)
    dataset = format_dataset(dataset, wrapped_model.get_tokenizer().eos_token)

    collator = DataCollatorForCompletionOnlyLM('AMR:', tokenizer=wrapped_model.get_tokenizer())

    trainer = SFTTrainer(
        model=wrapped_model.get_model(),
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        peft_config=get_peft_config(config) if config.use_lora else None,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        tokenizer=wrapped_model.get_tokenizer(),
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        # compute_metrics=wrapped_model.compute_metrics,
        data_collator=collator,
        args=get_training_arg(config),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.patience)]
    )
    trainer.train()

    trainer.save_model(os.path.join(config.output_dir, 'best_model'))

    shutil.copy2(config_path, Path(config.output_dir) / Path(config_path).name)


if __name__ == "__main__":
    main()
