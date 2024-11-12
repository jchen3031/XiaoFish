from call_meta import using_meta_model
# from tqdm import tqdm

with open(r'nl2bash/data/bash/all.cm', 'r', encoding="utf-8") as f:
    data = f.readlines()

print(data[:5])
res = using_meta_model(f"{data[0]}")
print(res)