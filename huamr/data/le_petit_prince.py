from torch.utils.data import Dataset


class LePetitPrinceDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._load_data()

    def __getitem__(self, index):
        x = self.data[index]

        return x

    def __len__(self):
        return len(self.data)

    def _load_data(self):
        amr_models = []

        with open(self.file_path, 'r') as f:
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
        train = self.data[:int(len(self.data) * 0.8)]
        dev = self.data[int(len(self.data) * 0.8):int(len(self.data) * 0.9)]
        test = self.data[int(len(self.data) * 0.9):]

        return train, dev, test
