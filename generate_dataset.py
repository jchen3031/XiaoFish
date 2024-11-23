from call_meta import using_meta_model
from tqdm import tqdm

with open(r'data/all.cm', 'r', encoding="utf-8") as f:
    data = f.readlines()

print(data[:5])

res = []
for i in tqdm(range(len(data))):
    res.append(using_meta_model(f'{data[i]}'))

with open("output.nl", "w", encoding="utf-8") as file:
    for element in data:
        file.write(element + "\n")
print(res[:5])