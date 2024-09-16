import os
import shutil
from pathlib import Path

import click
import pandas as pd
from datasets import DatasetDict, Dataset
from transformers import (
    IntervalStrategy,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from huamr.data.amr3 import AMR3Dataset
from huamr.s2s_models.base_model import S2SBaseModel
from huamr.utils.config_reader import get_config_from_yaml
from huamr.utils.langtype import LangType
from huamr.utils.model_factory import S2SModelFactory


def load_dataset(config, model: S2SBaseModel):
    dataset = AMR3Dataset(config.data_path, config.remove_wiki)
    train, validation, test = dataset.get_split(LangType[config.train_language], LangType[config.dev_language])
    dataset = DatasetDict({
        'train': Dataset.from_pandas(pd.DataFrame(train)),
        'validation': Dataset.from_pandas(pd.DataFrame(validation)),
        'test': Dataset.from_pandas(pd.DataFrame(test)),
    })

    dataset = dataset.map(model.process_data_to_model_inputs, batched=True, )

    return dataset


def get_training_arg(config):
    return Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        do_eval=True,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,

        log_level='debug',

        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,

        evaluation_strategy=IntervalStrategy.STEPS,
        predict_with_generate=True,
        save_steps=config.valid_steps,
        eval_steps=config.valid_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        logging_steps=config.logging_steps,

        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        group_by_length=config.group_by_length,
        generation_max_length=config.max_predict_length,
        generation_num_beams=config.num_beams,

        lr_scheduler_type='constant',
        fp16=config.fp16,
        report_to=None,
    )


@click.command()
@click.argument('config_path')
def main(config_path):
    config = get_config_from_yaml(config_path)

    wrapped_model: S2SBaseModel = S2SModelFactory.get_model(config)

    dataset = load_dataset(config, wrapped_model)

    trainer = Seq2SeqTrainer(
        model=wrapped_model.get_model(),
        args=get_training_arg(config),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.patience)],
        compute_metrics=wrapped_model.compute_metrics
    )

    trainer.train()

    trainer.model.save_pretrained(os.path.join(config.output_dir, 'best_model'))

    shutil.copy2(config_path, Path(config.output_dir) / Path(config_path).name)


if __name__ == "__main__":
    main()
