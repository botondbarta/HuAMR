import os
import shutil
from pathlib import Path

import click
import pandas as pd
from datasets import DatasetDict, Dataset
from dotmap import DotMap
from peft import LoraConfig, TaskType
from smatchpp import solvers, Smatchpp
from torch.cuda import is_available as cuda_available
from transformers import IntervalStrategy, AutoTokenizer, AutoModelForSeq2SeqLM
from trl.trainer import GRPOTrainer, GRPOConfig

from huamr.data.amr3 import AMR3Dataset
from huamr.entrypoints.train import calc_smatch_for_grpo
from huamr.llm_models.base_model import LLMBaseModel
from huamr.utils.amr_helper import strict_amr_check
from huamr.utils.amr_validator import AMRValidator
from huamr.utils.config_reader import get_config_from_yaml
from huamr.utils.langtype import LangType

HF_TOKEN = os.getenv('HF_TOKEN')

ilp = solvers.ILP()
measure = Smatchpp(alignmentsolver=ilp)


def load_dataset(config: DotMap) -> DatasetDict:
    dataset = AMR3Dataset(config.data_path, config.remove_wiki)
    train, validation, _ = dataset.get_split(LangType['EN'], LangType['EN'])

    train_df = pd.DataFrame(train)
    train_df = train_df.sample(frac=1).iloc[:config.gold_data_amount]

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(pd.DataFrame(validation)),
    })

    return dataset


def format_dataset(dataset: DatasetDict):
    def format_examples(examples):
        texts = [sentence for sentence in examples["sentence"]]

        return {"prompt": texts}

    return dataset.map(format_examples, batched=True)


def get_peft_config(config):
    return LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias='none',
        use_rslora=config.use_rslora,
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q", "v"]
    )


def get_training_config(config: DotMap) -> GRPOConfig:
    common_args = {
        "output_dir": config.output_dir,
        "do_eval": True,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "log_level": "debug",
        "optim": "paged_adamw_32bit",
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "eval_strategy": IntervalStrategy.STEPS,
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps,
        "save_total_limit": config.save_total_limit,
        "load_best_model_at_end": True,
        "logging_steps": config.logging_steps,
        "max_grad_norm": config.max_grad_norm,
        "num_train_epochs": config.num_train_epochs,
        "warmup_steps": config.warmup_steps,
        "group_by_length": config.group_by_length,
        "lr_scheduler_type": "linear",
        "report_to": "wandb",
    }

    return GRPOConfig(
        **common_args,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        num_generations=config.num_generations,
        remove_unused_columns=False,
        temperature=config.temperature,
        bf16=False,
    )


def reward_smatch(completions, **kwargs) -> list[float]:
    return [
        calc_smatch_for_grpo(comp, truth) if strict_amr_check(comp) else 0.0
        for comp, truth
        in zip(completions, kwargs['amr_graph'])
    ]


def reward_propbank_correctness(completions) -> list[float]:
    return [
        1.0
        if strict_amr_check(completion) and amr_validator.validate_against_propbank_frames(completion)
        else 0.0
        for completion in completions
    ]


def reward_and_or_connection(completions) -> list[float]:
    return [
        1.0
        if strict_amr_check(completion) and amr_validator.validate_and_or_connection(completion)
        else 0.0
        for completion in completions
    ]


def reward_amr_correctness(completions) -> list[float]:
    return [1.0 if strict_amr_check(completion) else 0.0 for completion in completions]


def create_trainer(wrapped_model: LLMBaseModel, dataset: DatasetDict, config: DotMap) -> GRPOTrainer:
    peft_config = get_peft_config(config) if config.use_lora else None
    model = wrapped_model.get_model()
    tokenizer = wrapped_model.get_tokenizer()
    formatted_dataset = format_dataset(dataset)

    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["validation"],
        peft_config=peft_config,
        reward_funcs=[reward_amr_correctness,
                      reward_smatch,
                      reward_propbank_correctness,
                      reward_and_or_connection],
        args=get_training_config(config),
    )


@click.command()
@click.argument('config_path')
def main(config_path):
    config = get_config_from_yaml(config_path)
    device = 'cuda' if cuda_available() else 'cpu'

    global amr_validator
    amr_validator = AMRValidator(config.frame_arg_descr)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name).to(device)

    dataset = load_dataset(config)
    formatted_dataset = format_dataset(dataset)

    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["validation"],
        peft_config=get_peft_config(config) if config.use_lora else None,
        reward_funcs=[reward_amr_correctness,
                      reward_smatch,
                      reward_propbank_correctness,
                      reward_and_or_connection],
        args=get_training_config(config),
    )

    trainer.train()
    trainer.save_model(os.path.join(config.output_dir, 'best_model'))

    shutil.copy2(config_path, Path(config.output_dir) / Path(config_path).name)


if __name__ == "__main__":
    main()
