import os
from pathlib import Path

import click
import chromadb
import pandas as pd
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig
from sentence_transformers import SentenceTransformer

from huamr.data.amr3 import AMR3Dataset
from huamr.utils.config_reader import get_config_from_yaml
from huamr.utils.langtype import LangType
from huamr.utils.model_factory import ModelFactory
from huamr.utils.amr_validator import AMRValidator


HF_TOKEN = os.getenv('HF_TOKEN')


def create_chromadb_collection(client, dataframe, collection_name, embedding_model, batch_size):
    sentences = dataframe['sentence'].tolist()
    amr_graphs = dataframe['generated_amr'].tolist()
    
    collection = client.create_collection(name=collection_name)

    batch_embeddings = []
    batch_sentences = []
    batch_metadatas = []
    batch_ids = []
    
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i: i + batch_size]
        batch_amr_graphs = amr_graphs[i: i + batch_size]
        
        batch_embeddings = embedding_model.encode(batch_sentences).tolist()
        batch_metadatas = [{'amr_graph': amr} for amr in batch_amr_graphs]
        batch_ids = [f"id{j}" for j in range(i, i + len(batch_sentences))]

        collection.add(
            embeddings=batch_embeddings,
            documents=batch_sentences,
            metadatas=batch_metadatas,
            ids=batch_ids
        )

    return collection

def load_existing_chromadb_collection(client, collection_name):
    return client.get_collection(name=collection_name)

def find_similar_examples(collection, target_sentence, n_examples, embedding_model):
    query_embedding = embedding_model.encode([target_sentence])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=n_examples)
        
    examples = [{"sentence": doc, "amr_graph": meta['amr_graph']} 
                for doc, meta in zip(results['documents'][0], results['metadatas'][0])]
    return examples

def batch_fewshot_inference(wrapped_model, sentences, collection, batch_size, n_examples, embedding_model):
    all_outputs = []
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i: i + batch_size]
        
        batch_prompts = [(find_similar_examples(collection, sentence, n_examples=n_examples, embedding_model=embedding_model), sentence)
                         for sentence in batch]
        batch_outputs = wrapped_model.fewshot_inference(batch_prompts)
        all_outputs.extend(batch_outputs)
    
    return all_outputs

def load_dataset(dataset_path):
    dataset = AMR3Dataset(dataset_path)
    _, _, test_set = dataset.get_split(test_lang=LangType['HU'])
    return pd.DataFrame(test_set)

@click.command()
@click.argument('config_path')
@click.argument('adapter_path')
@click.argument('output_path')
@click.argument('batch_size', type=int, default=32)
@click.argument('n_examples', type=int, default=1)
def main(config_path, adapter_path, output_path, batch_size, n_examples):
    config = get_config_from_yaml(config_path)

    embedding_model = SentenceTransformer(config.embedding_model)
    client = chromadb.PersistentClient(path=config.collection_path)
    amrvalidator = AMRValidator(config.frame_arg_descr)

    adapter = Path(adapter_path)
    wrapped_model = ModelFactory.get_model(config, HF_TOKEN)
    wrapped_model.model = PeftModel.from_pretrained(wrapped_model.get_model(), adapter)
    wrapped_model.model.eval()
    
    test_set = load_dataset(config.data_path)

    if config.collection_exists:
        collection = load_existing_chromadb_collection(client, config.collection_name)
    else:
        df = pd.read_csv(config.embedding_data_path)
        df = df[df.apply(lambda x: amrvalidator.validate(x['generated_amr']), axis=1)]
        collection = create_chromadb_collection(client, df, config.collection_name, embedding_model, batch_size)

    sentences = test_set['sentence'].tolist()
    generated_outputs = batch_fewshot_inference(
        wrapped_model, sentences, collection, batch_size, n_examples, embedding_model
    )
    
    test_set['generated_amr'] = [output.split('### AMR Graph')[-1].strip() for output in generated_outputs]
    test_set.to_csv(os.path.join(output_path, 'generated_fewshot.csv'), header=True, index=False)

if __name__ == '__main__':
    main()
