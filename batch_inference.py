import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import time
from toydataset import ToyDataset
# 设置随机种子
torch.random.manual_seed(0)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="fp4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_storage=torch.uint8
)


# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    "./phi3_mini",
    device_map="cuda:2",
    torch_dtype="auto",
    trust_remote_code=True,
    # quantization_config=bnb_config
    # load_in_4bit = True
)
tokenizer = AutoTokenizer.from_pretrained("./phi3_mini")

# 构建生成管道
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# 定义生成参数
generation_args = {
    "max_new_tokens": 220,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
    "use_cache": True,
}

# 创建数据集和数据加载器
file_path = 'toy.json'  # JSON 文件路径
dataset = ToyDataset(file_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)  # 使用batch_size为1

# 记录总耗时
total_time_start = time.time()

# 处理每个批次的数据
outputs = []
sample_count = 0
batch_times = []

for batch in dataloader:
    batch_inputs = [i for i in batch]  
    print(batch_inputs)
    
    _ = pipe(batch_inputs, **generation_args)
    batch_start_time = time.time()
    batch_outputs = pipe(batch_inputs, **generation_args)
    print(batch_outputs)
    batch_end_time = time.time()
    print(batch_end_time - batch_start_time)
    batch_times.append(batch_end_time - batch_start_time)
    outputs.extend(batch_outputs)
    sample_count += len(batch_inputs)

# 计算总时间和每个样本的时间
total_time_end = time.time()
total_time = total_time_end - total_time_start
# time_per_sample = total_time / sample_count if sample_count != 0 else 0
time_per_sample = sum(batch_times)/sample_count
total_time = sum(batch_times)

# 将生成的文本保存到 output.json 文件
with open("8bit_inference_output.json", "w", encoding="utf-8") as file, open("8bit-time_metrics.json", "w", encoding="utf-8") as time_file:
    json.dump(outputs, file, ensure_ascii=False, indent=4)
    json.dump({"total_time": total_time, "time_per_sample": time_per_sample}, time_file, ensure_ascii=False, indent=4)

# 打印时间指标
print(f"Total time for all samples: {total_time} s")
print(f"Time per sample: {time_per_sample} s")
