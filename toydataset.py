import json
from torch.utils.data import Dataset
# 定义自定义数据集
class ToyDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instruction = self.data[idx]["instruction"]
        # input_text = self.data[idx]["input"]
        # formatted_input = f"{instruction}"
        messages = [{"role": "user", "content": instruction}]
        return messages