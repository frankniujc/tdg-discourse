import torch

class DictDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def __getitem__(self, idx):
        item = {}
        for name, items in self.data_dict.items():
            item[name] = items[idx]
        return item

    def __len__(self):
        for name, item in self.data_dict.items():
            return len(item)
