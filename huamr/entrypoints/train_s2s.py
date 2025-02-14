import os
import shutil
from pathlib import Path

import click
from transformers import (
    IntervalStrategy,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from huamr.entrypoints.train import load_dataset
from huamr.s2s_models.base_model import S2SBaseModel
from huamr.utils.config_reader import get_config_from_yaml
from huamr.utils.model_factory import S2SModelFactory


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

        eval_strategy=IntervalStrategy.STEPS,
        predict_with_generate=True,
        save_steps=config.valid_steps,
        eval_steps=config.valid_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        logging_steps=config.logging_steps,
        max_grad_norm=config.max_grad_norm,

        num_train_epochs=config.num_train_epochs,
        warmup_steps=config.warmup_steps,
        group_by_length=config.group_by_length,
        generation_max_length=config.max_predict_length,
        generation_num_beams=config.num_beams,

        lr_scheduler_type='linear',
        fp16=config.fp16,
        report_to=None,
    )


@click.command()
@click.argument('config_path')
def main(config_path):
    config = get_config_from_yaml(config_path)

    wrapped_model: S2SBaseModel = S2SModelFactory.get_model(config)

    dataset = load_dataset(config)
    dataset = dataset.map(wrapped_model.process_data_to_model_inputs, batched=True, )

    trainer = Seq2SeqTrainer(
        model=wrapped_model.get_model(),
        args=get_training_arg(config),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.patience)],
    )

    trainer.train()

    trainer.model.save_pretrained(os.path.join(config.output_dir, 'best_model'))

    shutil.copy2(config_path, Path(config.output_dir) / Path(config_path).name)


if __name__ == "__main__":
    main()
