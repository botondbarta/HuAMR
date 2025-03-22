import gc
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import torch
import wandb
from datasets import Dataset, DatasetDict, load_dataset
from dotmap import DotMap
from huamr.data.amr3 import AMR3Dataset
from huamr.entrypoints.train import calc_smatch_for_grpo
from huamr.utils.config_reader import get_config_from_yaml
from torch import FloatTensor, LongTensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizer,
)


@dataclass
class TrainingArguments:
    epochs: int
    batch_size: int
    learning_rate: float
    update_old_after: int
    group_size: int
    logging_steps: int
    max_new_tokens: int
    max_document_length: int
    max_summary_length: int
    grpo_epsilon: float
    grpo_beta: float
    gradient_max_norm: float
    save_steps: int
    save_dir: str


@dataclass
class BatchRewards:
    rewards: FloatTensor


@dataclass
class GRPOOutput:
    loss: FloatTensor
    reward: FloatTensor
    kl: FloatTensor


def load_model(model_name: str) -> PreTrainedModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    )
    model = model.to(device)
    return model


def load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def load_dataset(config: DotMap) -> DatasetDict:
    dataset = AMR3Dataset(config.data_path, config.remove_wiki)
    train, validation, _ = dataset.get_split(LangType['EN'], LangType['EN'])

    train_df = pd.DataFrame(train)
    train_df = train_df.sample(frac=1).iloc[:config.gold_data_amount]

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(pd.DataFrame(validation)),
    })

    for split in dataset.column_names:
        dataset[split] = dataset[split].remove_columns(['id', 'split', 'lang'])
        dataset[split] = dataset[split].rename_columns(
            {"sentence": "document", "amr_graph": "summary"}
        )

    return dataset


def tokenize_dataset(
        dataset: DatasetDict,
        tokenizer: PreTrainedTokenizer,
        max_document_length: int,
        max_summary_length: int,
) -> DatasetDict:
    def tokenize_function(example):
        model_inputs = tokenizer(
            example["document"],
            max_length=max_document_length,
            truncation=True,
        )
        labels = tokenizer(
            example["summary"],
            max_length=max_summary_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(
        tokenize_function, batched=True, remove_columns=["document", "summary"]
    )


def rouge_reward(predictions: list[str], references: list[str]) -> float:
    """
    Calculate the average ROUGE (1, 2, Lsum) scores for a set of predictions and references.
    Args:
        predictions (list[str]): A list of predicted text strings.
        references (list[str]): A list of reference text strings.
    Returns:
        float: The average ROUGE score (ROUGE-1, ROUGE-2, and ROUGE-Lsum).
    """

    scores = rouge_eval.compute(predictions=predictions, references=references)
    return (scores["rouge1"] + scores["rouge2"] + scores["rougeLsum"]) / 3.0


def calc_smatch_for_grpo(comp_graph: str, ref_graph: str) -> float:
    try:
        comp_graph = PenmanDeSerializer(comp_graph).get_graph_string()
        ref_graph = PenmanDeSerializer(ref_graph).get_graph_string()
        score = measure.score_pair(comp_graph, ref_graph)['main']['F1'] / 100

        return score ** 2
    except Exception:
        return 0.0


def compute_rewards(
        token_ids: LongTensor, labels: LongTensor, tokenizer: PreTrainedTokenizer
) -> BatchRewards:
    """
    Compute rewards based on the ROUGE avg score between generated completions and reference summaries.

    Args:
        token_ids (LongTensor): Tensor containing token IDs of the generated completions.
        labels (LongTensor): Tensor containing token IDs of the reference summaries.
        tokenizer (PreTrainedTokenizer): Tokenizer used to decode the token IDs.

    Returns:
        BatchRewards: A tensor containing the computed rewards for each completion.
    """
    labels[labels == -100] = tokenizer.pad_token_id
    amr_gen = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    amr_ref = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rewards = []
    for gen, ref in zip(amr_gen, amr_ref):
        reward = 0.0

        is_valid = xfm_is_amr_valid(gen)

        if is_valid:
            reward += 1.0

            smatch = calc_smatch_for_grpo(gen, ref)
            reward += smatch

            graph_str = PenmanDeSerializer(gen).get_graph_string()
            if amr_validator.validate_against_propbank_frames(graph_str):
                reward += 1.0

            if amr_validator.validate_and_or_connection(graph_str):
                reward += 1.0

        rewards.append(reward)

    rewards = torch.tensor(rewards, device=token_ids.device)
    return BatchRewards(rewards)


def selective_log_softmax(
        logits: FloatTensor, index: LongTensor
) -> FloatTensor:
    """
    Computes the log softmax of the input logits selectively based on the provided indices.
    This function performs the same operation as applying `log_softmax` on the logits tensor
    along the last dimension and then gathering the results based on the provided indices.
    However, it processes the logits row by row to save memory by leveraging PyTorch internals.

    Taken from https://www.tylerromero.com/posts/2025-02-selective-log-softmax/

    Args:
        logits (FloatTensor): A tensor of shape (batch_size, num_classes) containing the raw
                              logits for each class.
        index (LongTensor): A tensor of shape (batch_size, num_indices) containing the indices
                            of the classes for which to compute the log softmax.

    Returns:
        FloatTensor: A tensor of shape (batch_size, num_indices) containing the log softmax
                     values for the specified indices.
    """

    token_logprobs = []
    for logits_row, index_row in zip(logits, index):
        logprobs_row = logits_row.log_softmax(dim=-1)
        token_logprobs_row = torch.gather(
            logprobs_row, dim=-1, index=index_row.unsqueeze(-1)
        ).squeeze(-1)
        token_logprobs.append(token_logprobs_row)
    return torch.stack(token_logprobs)


def gather_token_scores(logits: FloatTensor, generated_ids: LongTensor) -> FloatTensor:
    """
    Gathers token scores from logits based on generated token IDs.

    Args:
        logits (FloatTensor): The logits output from the model. It can be a tuple of tensors or a single tensor.
        generated_ids (LongTensor): The IDs of the generated tokens.

    Returns:
        FloatTensor: The token scores after applying a selective log softmax on the logits.
    """

    if isinstance(logits, tuple):
        # Stack the logits (batch_size*group_size, output_length, vocab)
        logits = torch.stack(logits, axis=0).permute((1, 0, 2))

    # Logsoftmax the logits
    token_scores = selective_log_softmax(logits, generated_ids)

    return token_scores


def compute_token_scores(
        model: PreTrainedModel,
        encoder_input_ids: LongTensor,
        encoder_attention_mask: LongTensor,
        decoder_input_ids: LongTensor,
        decoder_attention_mask: LongTensor,
        batch_size: int,
        group_size: int,
) -> FloatTensor:
    """
    Computes token scores for a given batch of input sequences using a pre-trained model.

    Args:
        model (PreTrainedModel): The pre-trained model to use for generating logits.
        encoder_input_ids (LongTensor): Tensor containing input IDs for the encoder.
        encoder_attention_mask (LongTensor): Tensor containing attention masks for the encoder inputs.
        decoder_input_ids (LongTensor): Tensor containing input IDs for the decoder.
        decoder_attention_mask (LongTensor): Tensor containing attention masks for the decoder inputs.
        batch_size (int): The size of the batch.
        group_size (int): The size of the group.

    Returns:
        FloatTensor: A tensor containing the computed token scores, reshaped to (batch_size, group_size, -1).
    """
    logits = model(
        input_ids=encoder_input_ids,
        attention_mask=encoder_attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
    ).logits
    scores = gather_token_scores(logits[:, :-1], decoder_input_ids[:, 1:])
    scores = scores.view(batch_size, group_size, -1)
    del logits
    torch.cuda.empty_cache()
    return scores


def grpo(
        generated_ids: LongTensor,
        old_scores: FloatTensor,
        current_scores: FloatTensor,
        reference_scores: FloatTensor,
        labels: LongTensor,
        tokenizer: PreTrainedTokenizer,
        epsilon: float,
        beta: float,
) -> GRPOOutput:
    """
    Compute the loss of Group Relative Policy Optimization (GRPO) on the given inputs.

    Args:
        generated_ids (LongTensor): Tensor of generated token IDs.
        old_scores (FloatTensor): Tensor of old policy scores.
        current_scores (FloatTensor): Tensor of current policy scores.
        reference_scores (FloatTensor): Tensor of reference policy scores.
        truths (LongTensor): Tensor of ground truth token IDs.
        tokenizer (PreTrainedTokenizer): Tokenizer used for encoding/decoding.
        epsilon (float): Clipping parameter for policy ratios.
        beta (float): Weighting factor for the Kullback-Leibler divergence term.

    Returns:
        GRPOOutput: A dataclass containing the mean loss, rewards and KL divergences.
    """
    losses = torch.zeros(generated_ids.shape[0])
    rewards = torch.zeros(generated_ids.shape[0])
    kls = torch.zeros(generated_ids.shape[0])

    for idx, (
            group_ids,
            group_labels,
            group_old_scores,
            group_current_scores,
            group_reference_scores,
    ) in enumerate(
        zip(generated_ids, labels, old_scores, current_scores, reference_scores)
    ):
        # Compute advantages
        group_rewards = compute_rewards(group_ids, group_labels, tokenizer)
        mean = group_rewards.rewards.mean()
        centered = group_rewards.rewards - mean
        std = group_rewards.rewards.std()
        if std < 1e-8:
            advantages = torch.zeros_like(centered)
        else:
            advantages = centered / (std + 1e-8)

        # Store the mean of each rewards for the group
        rewards[idx] = group_rewards.rewards.mean()

        # Compute the ratios
        ratios = torch.exp(group_current_scores - group_old_scores)

        # Compute the clipped ratios
        clipped_ratios = torch.clamp(
            ratios, min=1.0 - epsilon, max=1.0 + epsilon
        )

        # Compute kullback-leibler divergence between reference and current policy
        kl = (
                torch.exp(group_reference_scores - group_current_scores)
                - (group_reference_scores - group_current_scores)
                - 1
        )
        kls[idx] = kl.mean()

        # Compute mean loss of the group
        completion_mask = group_ids[:, 1:] != tokenizer.pad_token_id
        loss = (
                torch.min(
                    ratios * advantages.unsqueeze(-1),
                    clipped_ratios * advantages.unsqueeze(-1),
                )
                - beta * kl
        )
        loss = -(loss * completion_mask).sum() / completion_mask.sum()
        losses[idx] = loss

    return GRPOOutput(
        loss=losses.mean(),
        reward=rewards.mean(),
        kl=kls.mean(),
    )


def train(
        dataset: Dataset,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        training_args: TrainingArguments,
) -> None:
    """
    Train a language model using the GRPO (Group Relative Policy Optimization) objective.

    Args:
        dataset (Dataset): The dataset containing training data.
        model (PreTrainedModel): The model to be trained.
        tokenizer (PreTrainedTokenizer): The tokenizer used for encoding the data.
        training_args (TrainingArguments): The training arguments containing hyperparameters and configurations.
    """
    # Prepare the dataloader
    train_dataloader = DataLoader(
        dataset["train"],
        collate_fn=DataCollatorForSeq2Seq(tokenizer),
        batch_size=training_args.batch_size,
    )

    # Prepare policies
    reference_model = deepcopy(model)
    old_model = deepcopy(model)
    reference_model.eval()
    old_model.eval()
    model.train()

    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

    scheduler = LinearLR(
        optimizer,
        start_factor=1,
        end_factor=0.1,
        total_iters=training_args.epochs * len(train_dataloader),
    )

    running_metrics = {
        "loss": 0.0,
        "reward": 0.0,
        "completion_length": 0.0,
        "kl": 0.0,
    }

    training_step = 0
    best_reward = 0.0
    for _ in range(training_args.epochs):
        # Update the old policy
        old_model.load_state_dict(model.state_dict(), strict=False)
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            # Prepare the batch data
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            effective_batch_size = input_ids.shape[0]

            # Generate ids with the old policy
            generated_ids = old_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=training_args.max_new_tokens,
                do_sample=True,
                num_beams=training_args.group_size,
                num_return_sequences=training_args.group_size,
            )

            # Prepare attention mask for computing current
            # and reference logits on the generated ids
            decoder_attention_mask = generated_ids != tokenizer.pad_token_id

            # Interleave input_ids and attention_mask to have
            # the same shape than the generated completions
            repeated_input_ids = input_ids.repeat_interleave(
                repeats=training_args.group_size, dim=0
            )
            repeated_attention_mask = attention_mask.repeat_interleave(
                repeats=training_args.group_size, dim=0
            )

            # Compute the sequence scores of the old policy
            with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
            ):
                old_scores = compute_token_scores(
                    old_model,
                    encoder_input_ids=repeated_input_ids,
                    encoder_attention_mask=repeated_attention_mask,
                    decoder_input_ids=generated_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    batch_size=effective_batch_size,
                    group_size=training_args.group_size,
                )

            # Compute the sequence scores of the current policy
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model.eval()
                current_scores = compute_token_scores(
                    model,
                    encoder_input_ids=repeated_input_ids,
                    encoder_attention_mask=repeated_attention_mask,
                    decoder_input_ids=generated_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    batch_size=effective_batch_size,
                    group_size=training_args.group_size,
                )
                model.train()

            # Compute the sequence scores of the reference model
            with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
            ):
                reference_scores = compute_token_scores(
                    reference_model,
                    encoder_input_ids=repeated_input_ids,
                    encoder_attention_mask=repeated_attention_mask,
                    decoder_input_ids=generated_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    batch_size=effective_batch_size,
                    group_size=training_args.group_size,
                )

            # Group the generated ids (batch_size, group_size, output_length)
            generated_ids = generated_ids.view(
                effective_batch_size, training_args.group_size, -1
            )

            # Repeat the labels and group (batch_size, group_size)
            labels = labels.repeat_interleave(
                repeats=training_args.group_size, dim=0
            ).view(effective_batch_size, training_args.group_size, -1)

            # Compute GRPO objective
            with torch.autocast("cuda", dtype=torch.bfloat16):
                grpo_output = grpo(
                    generated_ids,
                    old_scores,
                    current_scores,
                    reference_scores,
                    labels,
                    tokenizer,
                    training_args.grpo_epsilon,
                    training_args.grpo_beta,
                )

            # Update the current policy
            grpo_output.loss.backward()
            clip_grad_norm_(
                model.parameters(),
                training_args.gradient_max_norm,
            )
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # Update old policy periodically
            if (training_step + 1) % training_args.update_old_after == 0:
                old_model.load_state_dict(model.state_dict(), strict=False)
                torch.cuda.empty_cache()

            # Update log metrics
            batch_metrics = {
                "loss": grpo_output.loss.item(),
                "reward": grpo_output.reward.item(),
                "kl": grpo_output.kl.item(),
                "completion_length": decoder_attention_mask.sum(-1)
                .float()
                .mean()
                .item(),
            }
            running_metrics = {
                key: running_metrics[key] + batch_metrics.get(key, 0)
                for key in running_metrics
            }

            # And report them periodically
            if (training_step + 1) % training_args.logging_steps == 0:
                wandb.log(
                    {
                        **{
                            key: val / (training_step + 1)
                            for key, val in running_metrics.items()
                        },
                        **{"lr": scheduler.get_last_lr()[0]},
                    }
                )

            # Save the model each periodically
            if (training_step + 1) % training_args.save_steps == 0:
                last_reward = running_metrics["loss"] / (training_step + 1)
                if last_reward > best_reward:
                    model.save_pretrained(f"{training_args.save_dir}")
                    best_reward = last_reward
                    print(
                        "Saving model with reward:",
                        best_reward,
                        f"step: {training_step + 1}",
                    )
                else:
                    print(
                        f"Model not saved because didn't improve the reward at step {training_step + 1}"
                    )

            # Free GPU memory at the end
            del (
                generated_ids,
                old_scores,
                input_ids,
                attention_mask,
                repeated_input_ids,
                repeated_attention_mask,
                current_scores,
                reference_scores,
                grpo_output,
                labels,
            )
            torch.cuda.empty_cache()
            gc.collect()
            training_step += 1


@click.command()
@click.argument('config_path')
def main(config_path):
    config = get_config_from_yaml(config_path)

    model_name = '/mnt/idms/home/botondbarta/models/model_parse_xfm_bart_large-v0_1_0'
    model = load_model(model_name)
    tokenizer = load_tokenizer('facebook/bart-large')

    training_args = TrainingArguments(
        epochs=config.num_train_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        update_old_after=1000,
        group_size=config.num_generations,
        logging_steps=config.logging_steps,
        max_new_tokens=config.max_completion_length,
        max_document_length=config.max_prompt_length,
        max_summary_length=config.max_completion_length,
        grpo_epsilon=0.1,
        grpo_beta=0.04,
        gradient_max_norm=config.max_grad_norm,
        save_steps=config.save_steps,
        save_dir="/home/botondbarta/experiments/amr/xfm_grpo",
    )

    dataset = load_dataset(config)

    dataset = tokenize_dataset(
        dataset,
        tokenizer,
        training_args.max_document_length,
        training_args.max_summary_length,
    )

    wandb.init(
        project="GRPO-AMR",
        config={
            "model": model_name,
            "training_args": training_args.__dict__,
        },
    )

    train(dataset, model, tokenizer, training_args)

    # Save the model and finish logging
    model.save_pretrained(f"/home/botondbarta/experiments/amr/xfm_grpo/xfm_grpo_amr_model")
    wandb.finish()


if __name__ == "__main__":
    main()
