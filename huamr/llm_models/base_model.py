from abc import ABC

import numpy as np
from dotmap import DotMap
from peft import prepare_model_for_kbit_training
from smatchpp import Smatchpp, solvers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from huamr.utils import get_bnb_config


class LLMBaseModel(ABC):
    def __init__(self, config: DotMap, hf_token, do_train: bool = False):
        self.config = config
        self.hf_token = hf_token

        self.model = self.load_model(config.model_name, config.quantize, hf_token)
        self.tokenizer = self.load_tokenizer(config.model_name, config.hf_token)

        self.set_special_tokens(config.model_name, self.tokenizer)

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.use_cache = False

        if config.quantize and do_train:
            self.model = prepare_model_for_kbit_training(self.model)

        ilp = solvers.ILP()
        self.measure = Smatchpp(alignmentsolver=ilp)

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def load_model(self, model_name, quantize, hf_token):
        return AutoModelForCausalLM.from_pretrained(model_name,
                                                    quantization_config=get_bnb_config(quantize) if quantize else None,
                                                    device_map='auto',
                                                    token=hf_token)

    def load_tokenizer(self, model_name, hf_token):
        return AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side='left', token=hf_token)

    def inference(self, sentences: list[str]) -> list[str]:
        prompts = [f"### Instruction\nProvide the AMR graph for the following sentence. Ensure that the graph captures the main concepts, the relationships between them, and any additional information that is important for understanding the meaning of the sentence. Use standard AMR notation, including concepts, roles, and relationships.\n\n### Sentence\n{sentence}\n\n### AMR Graph\n" for sentence in sentences]
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")

        generation_config = GenerationConfig(
            num_beams=5,
            return_dict_in_generate=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=self.config.generate_max_length,
        )
        outputs = self.model.generate(**inputs, generation_config=generation_config)
        return self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=True)
        
    def fewshot_inference(self, batch_prompts) -> list[str]:
        prompts = []
        for examples, target_sentence in batch_prompts:

            prompt = "### Instruction\n" \
                     "Provide the AMR graph for the following sentence. Ensure that the graph captures " \
                     "the main concepts, the relationships between them, and any additional information " \
                     "that is important for understanding the meaning of the sentence. Use standard AMR " \
                     "notation, including concepts, roles, and relationships.\n\n"
            
            for example in examples:
                prompt += f"### Sentence\n{example['sentence']}\n\n### AMR Graph\n{example['amr_graph']}\n\n"
            
            prompt += f"### Sentence\n{target_sentence}\n\n### AMR Graph\n"
            prompts.append(prompt)

        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")
        
        generation_config = GenerationConfig(
            return_dict_in_generate=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=self.config.generate_max_length,
        )
        outputs = self.model.generate(**inputs, generation_config=generation_config)
        return self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def set_special_tokens(self, model_name, tokenizer):
        pass

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        # Decode generated summaries into text
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.split('\nAMR: ')[-1].strip() for pred in decoded_preds]
        decoded_labels = [label.split('\nAMR: ')[-1].strip() for label in decoded_labels]

        smatch_score, _ = self.measure.score_corpus(decoded_labels, decoded_preds)

        smatch_f1 = smatch_score['main']['F1']['result']
        smatch_prec = smatch_score['main']['Precision']['result']
        smatch_rec = smatch_score['main']['Recall']['result']

        return {
            'smatch_f1': smatch_f1,
            'smatch_prec': smatch_prec,
            'smatch_rec': smatch_rec,
        }
