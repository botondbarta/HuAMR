import copy
import os
import re

import pandas as pd
import penman

from huamr.utils.langtype import LangType


class AMR3Dataset:
    def __init__(self, folder_path, remove_wiki: bool = True):
        self.folder_path = folder_path
        self.data = self._load_data(folder_path, remove_wiki)

    def _load_data(self, folder_path, remove_wiki: bool = True):
        translations = self._load_translations(folder_path, 'translation')
        all_data_df = self._load_all_annotated_data(folder_path)

        df = pd.merge(all_data_df, translations, on='id', how='left')

        df_en_sentence = df[['id', 'sentence', 'amr_graph', 'split']].copy()
        df_hu_sentence = df[['id', 'hu_sentence', 'amr_graph', 'split']].copy()
        df_hu_sentence = df_hu_sentence.rename(columns={'hu_sentence': 'sentence'})
        df_hu_sentence = df_hu_sentence.dropna(subset=['sentence'])

        df_en_sentence['lang'] = LangType.EN.value
        df_hu_sentence['lang'] = LangType.HU.value

        df = pd.concat([df_en_sentence, df_hu_sentence], ignore_index=True)

        df = df.drop_duplicates(subset='sentence')

        df['penman_graph'] = df['amr_graph'].apply(penman.decode)
        if remove_wiki:
            df['penman_graph'] = df['penman_graph'].apply(self.remove_wiki_from_graph)
        df['amr_graph'] = df['penman_graph'].apply(self._linearize)

        return df[['id', 'sentence', 'amr_graph', 'split', 'lang']]

    def _load_all_annotated_data(self, folder_path):
        train = self._load_data_from_folder(folder_path, 'training')
        dev = self._load_data_from_folder(folder_path, 'dev')
        test = self._load_data_from_folder(folder_path, 'test')

        return pd.concat([train, dev, test], ignore_index=True)

    def _load_data_from_folder(self, folder_path, split):
        folder_path = os.path.join(folder_path, split)
        amr_models = []

        for file in os.listdir(folder_path):
            with open(os.path.join(folder_path, file), 'r') as f:
                lines = f.readlines()

            save_current = False
            current_id = None
            current_sentence = None
            current_amr_graph = []
            is_reading_amr_graph = False

            for line in lines:
                if '# ::id' in line:
                    current_id = line.split('::id')[-1].strip().split('::date')[0].strip()

                elif '# ::snt' in line:
                    current_sentence = line.split('::snt')[-1].strip()
                    current_amr_graph = []
                elif line.startswith('('):
                    is_reading_amr_graph = True
                    current_amr_graph.append(line)
                elif is_reading_amr_graph:
                    if line.strip() == '':
                        is_reading_amr_graph = False
                        save_current = True
                    else:
                        current_amr_graph.append(line)

                if save_current:
                    amr_models.append({
                        'id': current_id,
                        'sentence': current_sentence,
                        'amr_graph': ''.join(current_amr_graph).strip(),
                        'split': split,
                    })
                    save_current = False

        return pd.DataFrame(amr_models)

    def _load_translations(self, folder_path, translation_folder) -> pd.DataFrame:
        folder_path = os.path.join(folder_path, translation_folder)

        translations = []
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                translations.append(pd.read_csv(os.path.join(folder_path, file)))

        df = pd.concat(translations, ignore_index=True)

        df.drop('sentence', axis=1, inplace=True)
        return df

    def remove_wiki_from_graph(self, graph: penman.Graph) -> penman.Graph:
        # https://github.com/BramVanroy/multilingual-text-to-amr/blob/efc1f7249bda34cd01dbe3ced2deaa5edeff84b8/src/multi_amr/utils/__init__.py#L79
        # modified from SPRING
        triples = []
        for t in graph.triples:
            v1, rel, v2 = t
            if rel == ":wiki":
                t = penman.Triple(v1, rel, "+")
            triples.append(t)

        return penman.Graph(triples, metadata=graph.metadata)

    def _linearize(self, graph: penman.Graph) -> str:
        # https://github.com/BramVanroy/multilingual-text-to-amr/blob/main/src/multi_amr/tokenization.py#L329
        # modified from SPRING
        graph_ = copy.deepcopy(graph)
        graph_.metadata = {}
        try:
            linearized = penman.encode(graph_).replace("–", "-")  # NLLB does not have an en-hyphen
        except Exception as exc:
            print(graph_)
            print(graph_.metadata)
            raise exc

        linearized_nodes = self._tokenize_encoded_graph(linearized)
        # remap = {}
        # for i in range(1, len(linearized_nodes)):
        #     nxt = linearized_nodes[i]
        #     lst = linearized_nodes[i - 1]
        #     if nxt == "/":
        #         remap[lst] = f"<pointer:{len(remap)}>"

        # i = 1
        # linearized_nodes_ = [linearized_nodes[0]]
        # while i < (len(linearized_nodes)):
        #     nxt = linearized_nodes[i]
        #     lst = linearized_nodes_[-1]
        #     if nxt in remap:
        #         if lst == "(" and linearized_nodes[i + 1] == "/":
        #             nxt = remap[nxt]
        #             i += 1
        #         elif lst.startswith(":"):
        #             nxt = remap[nxt]
        #     elif lst == ":polarity" and nxt == "-":
        #         linearized_nodes_[-1] = ":negation"
        #         i += 1
        #         continue
        #     linearized_nodes_.append(nxt)
        #     i += 1

        # linearized_nodes_ = [tstrip for t in linearized_nodes_ if (tstrip := t.strip())]
        linearized_nodes_ = [tstrip for t in linearized_nodes if (tstrip := t.strip())]
        linearized_graph = ' '.join(linearized_nodes_)
        return linearized_graph

    def _tokenize_encoded_graph(self, linearized: str) -> list[str]:
        # https://github.com/BramVanroy/multilingual-text-to-amr/blob/main/src/multi_amr/tokenization.py#L370
        # modified from SPRING
        linearized = re.sub(r"(\".+?\")", r" \1 ", linearized)
        pieces = []
        for piece in linearized.split():
            if piece.startswith('"') and piece.endswith('"'):
                pieces.append(piece)
            else:
                piece = piece.replace("(", " ( ")
                piece = piece.replace(")", " ) ")
                piece = piece.replace(":", " :")
                piece = piece.replace("/", " / ")
                piece = piece.strip()
                pieces.append(piece)
        linearized = re.sub(r"\s+", " ", " ".join(pieces)).strip()
        return linearized.split(" ")

    def get_split(
            self,
            train_lang: LangType = LangType.ALL,
            dev_lang: LangType = LangType.ALL,
            test_lang: LangType = LangType.ALL,
    ):
        def filter_by_lang(data, lang_type):
            if lang_type == LangType.ALL:
                return data
            return data[data['lang'] == lang_type.value]

        train = filter_by_lang(self.data[self.data['split'] == 'training'], train_lang).to_dict(orient='records')
        dev = filter_by_lang(self.data[self.data['split'] == 'dev'], dev_lang).to_dict(orient='records')
        test = filter_by_lang(self.data[self.data['split'] == 'test'], test_lang).to_dict(orient='records')

        return train, dev, test
