from datasets import load_dataset

from quntoken import tokenize
from tqdm import tqdm


tqdm.pandas()

def sentence_tokenize(text: str):
    doc = tokenize(text, mode='sentence')
    return [
        sent.replace('\n', '').replace('\\n', '').replace('""', '').replace('" "', '')
        for sent in doc
        if sent != '\n'
    ]

def main():
    ds = load_dataset("SZTAKI-HLT/HunSum-2-abstractive")
    ds = ds['train']
    ds = ds.remove_columns(['date_of_creation', 'tags', 'article', 'title'])
    df = ds.to_pandas()

    df = df[df['domain'] == 'index.hu']
    df = df.sample(n=50000, random_state=42)

    df['lead_sentences'] = df['lead'].progress_apply(sentence_tokenize)

    df['sentence'] = df['lead_sentences'].apply(lambda x: x[0])
    df = df[['uuid', 'url', 'sentence']]

    df.to_csv('synthetic_data_input.csv', index=False, header=True)

if __name__ == '__main__':
    main()