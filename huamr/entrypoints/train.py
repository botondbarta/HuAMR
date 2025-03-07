import os
import shutil
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from datasets import DatasetDict, Dataset
from dotmap import DotMap
from peft import LoraConfig
from smatchpp import solvers, Smatchpp
from transformers import IntervalStrategy, EarlyStoppingCallback
from trl.trainer import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig

from huamr.data.amr3 import AMR3Dataset
from huamr.llm_models.base_model import LLMBaseModel
from huamr.utils.amr_helper import strict_amr_check
from huamr.utils.amr_validator import AMRValidator
from huamr.utils.config_reader import get_config_from_yaml
from huamr.utils.constants import SYSTEM_PROMPT
from huamr.utils.langtype import LangType
from huamr.utils.model_factory import ModelFactory

HF_TOKEN = os.getenv('HF_TOKEN')

ilp = solvers.ILP()
measure = Smatchpp(alignmentsolver=ilp)


def load_synthetic_data(file, synthetic_data_amount) -> Optional[pd.DataFrame]:
    if file:
        df = pd.read_csv(file)
        df = df.rename(columns={'generated_amr': 'amr_graph'})
        df = df.iloc[:synthetic_data_amount]
        return df

    return None


def load_dataset(config: DotMap) -> DatasetDict:
    train_df = pd.DataFrame()

    dataset = AMR3Dataset(config.data_path, config.remove_wiki)
    train, validation, _ = dataset.get_split(LangType[config.train_language], LangType[config.dev_language])

    if config.load_amr3:
        train_df = pd.DataFrame(train)
        train_df = train_df.sample(frac=1).iloc[:config.gold_data_amount]

    synthetic_data = load_synthetic_data(config.synthetic_data, config.synthetic_data_amount)

    concatenated = pd.concat([train_df, synthetic_data])
    concatenated = concatenated.sample(frac=1).reset_index(drop=True)

    dataset = DatasetDict({
        'train': Dataset.from_pandas(concatenated),
        'validation': Dataset.from_pandas(pd.DataFrame(validation)),
    })

    return dataset


def format_dataset(dataset: DatasetDict, tokenizer, training_method: str):
    def format_examples(examples):
        sentences = examples["sentence"]
        amr_graphs = examples["amr_graph"]
        texts = []

        for sentence, amr_graph in zip(sentences, amr_graphs):
            chat_temp = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sentence},
            ]

            if training_method == "sft":
                chat_temp.append({"role": "assistant", "content": amr_graph})

            texts.append(chat_temp)

        return {"prompt": texts}

    return dataset.map(format_examples, batched=True)


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


def get_training_config(config: DotMap, training_method: str) -> SFTConfig | GRPOConfig:
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

    if training_method == "sft":
        return SFTConfig(
            **common_args,
            dataset_text_field="prompt",
            max_seq_length=config.max_seq_length,
            bf16=True,
        )
    elif training_method == "grpo":
        return GRPOConfig(
            **common_args,
            max_prompt_length=config.max_prompt_length,
            max_completion_length=config.max_completion_length,
            num_generations=config.num_generations,
            remove_unused_columns=False,
            temperature=config.temperature,
            bf16=False,
        )
    else:
        raise ValueError(f"Unsupported training method: {training_method}")


def calc_smatch_for_grpo(comp_graph: str, ref_graph: str) -> float:
    try:
        score = measure.score_pair(comp_graph, ref_graph)['main']['F1'] / 100

        return score ** 2
    except Exception:
        return 0.0


def reward_smatch(completions, **kwargs) -> list[float]:
    completions = [completion[0]["content"] for completion in completions]

    return [
        calc_smatch_for_grpo(comp, truth) if strict_amr_check(comp) else 0.0
        for comp, truth
        in zip(completions, kwargs['amr_graph'])
    ]


def reward_propbank_correctness(completions, **kwargs) -> list[float]:
    completions = [completion[0]["content"] for completion in completions]

    return [
        1.0
        if strict_amr_check(completion) and amr_validator.validate_against_propbank_frames(completion)
        else 0.0
        for completion in completions
    ]


def reward_and_or_connetion(completions, **kwargs) -> list[float]:
    completions = [completion[0]["content"] for completion in completions]

    return [
        1.0
        if strict_amr_check(completion) and amr_validator.validate_and_or_connection(completion)
        else 0.0
        for completion in completions
    ]


def reward_amr_correctness(completions, **kwargs) -> list[float]:
    completions = [completion[0]["content"] for completion in completions]
    return [1.0 if strict_amr_check(completion) else 0.0 for completion in completions]


def create_trainer(wrapped_model: LLMBaseModel, dataset: DatasetDict, config: DotMap,
                   training_method: str) -> SFTTrainer | GRPOTrainer:
    peft_config = get_peft_config(config) if config.use_lora else None
    model = wrapped_model.get_model()
    tokenizer = wrapped_model.get_tokenizer()
    formatted_dataset = format_dataset(dataset, tokenizer, training_method)

    if training_method == "sft":
        return SFTTrainer(
            model=model,
            train_dataset=formatted_dataset["train"],
            eval_dataset=formatted_dataset["validation"],
            peft_config=peft_config,
            processing_class=tokenizer,
            args=get_training_config(config, training_method),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config.patience)]
        )
    elif training_method == "grpo":
        return GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=formatted_dataset["train"],
            eval_dataset=formatted_dataset["validation"],
            peft_config=peft_config,
            reward_funcs=[reward_amr_correctness,
                          reward_smatch,
                          reward_propbank_correctness,
                          reward_and_or_connetion],
            args=get_training_config(config, training_method),
        )
    else:
        raise ValueError(f"Unsupported training method: {training_method}")


@click.command()
@click.argument('config_path')
@click.argument('training_method', default='sft', type=click.Choice(['sft', 'grpo']))
def main(config_path, training_method):
    config = get_config_from_yaml(config_path)

    global amr_validator
    amr_validator = AMRValidator(config.frame_arg_descr)

    wrapped_model = ModelFactory.get_model(config, HF_TOKEN, do_train=True)

    dataset = load_dataset(config)
    trainer = create_trainer(wrapped_model, dataset, config, training_method)

    trainer.train()
    trainer.save_model(os.path.join(config.output_dir, 'best_model'))

    shutil.copy2(config_path, Path(config.output_dir) / Path(config_path).name)


if __name__ == "__main__":
    main()
