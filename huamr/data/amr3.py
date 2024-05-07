import os

import pandas as pd


class AMR3Dataset:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.data = self._load_data(folder_path)

    def _load_data(self, folder_path):
        translations = self._load_translations(folder_path, 'translation')
        all_data_df = self._load_all_annotated_data(folder_path)

        df = pd.merge(all_data_df, translations, on='id', how='left')

        df_sentence = df[['id', 'sentence', 'amr_graph', 'split']].copy()
        df_hu_sentence = df[['id', 'hu_sentence', 'amr_graph', 'split']].copy()
        df_hu_sentence = df_hu_sentence.rename(columns={'hu_sentence': 'sentence'})
        df_hu_sentence = df_hu_sentence.dropna(subset=['sentence'])

        return pd.concat([df_sentence, df_hu_sentence], ignore_index=True)

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

    def _load_translations(self, folder_path, translation_folder):
        folder_path = os.path.join(folder_path, translation_folder)

        translations = []
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                translations.append(pd.read_csv(os.path.join(folder_path, file)))

        df = pd.concat(translations, ignore_index=True)

        df.drop('sentence', axis=1, inplace=True)
        return df

    def get_split(self):
        train = self.data[self.data['split'] == 'training'].to_dict(orient='records')
        dev = self.data[self.data['split'] == 'dev'].to_dict(orient='records')
        test = self.data[self.data['split'] == 'test'].to_dict(orient='records')

        return train, dev, test
