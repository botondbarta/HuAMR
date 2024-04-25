import os


class AMR3Dataset:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.train = self._load_data(os.path.join(folder_path, 'training'))
        self.dev = self._load_data(os.path.join(folder_path, 'dev'))
        self.test = self._load_data(os.path.join(folder_path, 'test'))

    def _load_data(self, folder_path):
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
                        'amr_graph': ''.join(current_amr_graph).strip()
                    })
                    save_current = False

        return amr_models

    def get_split(self):
        return self.train, self.dev, self.test
