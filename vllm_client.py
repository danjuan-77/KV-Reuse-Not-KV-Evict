from openai import OpenAI
import time
import json
from torch.utils.data import DataLoader
from toydataset import ToyDataset

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "token-abc123"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# 创建数据集和数据加载器
file_path = 'toy.json'  # JSON 文件路径
dataset = ToyDataset(file_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)  # 使用batch_size为1

def chat(message):
    """ chat 模式
    """
    time1 = time.time()
    completion = client.chat.completions.create(
      model="./phi3_mini",
      messages=[message]
    )
    time2 = time.time()
    inference_time = time2 - time1
    response_message = completion.choices[0].message['content'] if isinstance(completion.choices[0].message, dict) else completion.choices[0].message.content  # 提取实际的消息内容
    return response_message, inference_time

if __name__ == "__main__":
    total_time = 0
    outputs = []

    for batch in dataloader:
        message = batch[0][0]
        response, inference_time = chat(message)
        outputs.append({
            "input": message,
            "output": response,
            "inference_time": inference_time
        })
        total_time += inference_time

    average_time = total_time / len(outputs)

    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average inference time per sample: {average_time:.4f} seconds")

    with open('vllm outputs.json', 'w') as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)
