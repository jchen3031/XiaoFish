import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from XiaoFishBot import Transformer, CustomTransformer, CustomSchedule
from utils import predict_sequence_step_by_step
# 定义保存的路径
model_dir = "model_2"
tokenizer_path = os.path.join(model_dir, "tokenizers")
saved_model_path = os.path.join(model_dir, "transformer_model")

# 加载模型
loaded_model = load_model(saved_model_path, custom_objects={
    "Transformer": Transformer,
    "CustomTransformer": CustomTransformer,
    "CustomSchedule": CustomSchedule
})

print("Model loaded successfully!")

# 加载分词器
with open(os.path.join(tokenizer_path, 'source_tokenizer.pickle'), 'rb') as handle:
    source_tokenizer = pickle.load(handle)

with open(os.path.join(tokenizer_path, 'target_tokenizer.pickle'), 'rb') as handle:
    target_tokenizer = pickle.load(handle)

print("Tokenizers loaded successfully!")
input_sentence = "Make directory '/cpuset'"
predicted_translation = predict_sequence_step_by_step(input_sentence, loaded_model, source_tokenizer, target_tokenizer)
print(f"Input: {input_sentence}")
print(f"Predicted Translation: {predicted_translation}")
from utils import get_bleu
import re
print("GPU Available:", tf.config.list_physical_devices('GPU'))
# 读取文件内容
with open(r'data/all.nl', 'r', encoding='utf-8') as f_cm:
    source_sentences = f_cm.readlines()

with open(r'data/all.cm', 'r', encoding='utf-8') as f_nl:
    target_sentences = f_nl.readlines()

#filters = r'!"#$%&()*+,;=?@[\\]^`{|}~'
filters = r'!"#$%&()*+,;=?@[\\]^_`{|}~'
filters = filters.replace('|', '')


def preprocess_text(text):
    text = re.sub(r'[\t\n]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


source_sentences = [preprocess_text(s) for s in source_sentences]
target_sentences = [preprocess_text(s) for s in target_sentences]
input_sequences = source_tokenizer.texts_to_sequences(source_sentences)
target_sequences = target_tokenizer.texts_to_sequences(target_sentences)
BUFFER_SIZE = 20000
data_size = len(input_sequences)
train_size = int(0.8 * data_size)
val_size = int(0.1 * data_size)
test_size = data_size - train_size - val_size

# 划分数据集
train_input = input_sequences[:train_size]
train_target = target_sequences[:train_size]

val_input = input_sequences[train_size:train_size + val_size]
val_target = target_sequences[train_size:train_size + val_size]

test_input = input_sequences[train_size + val_size:]
test_target = target_sequences[train_size + val_size:]
from tensorflow.keras.preprocessing.sequence import pad_sequences
# 填充序列
train_input = pad_sequences(train_input, padding='post')
train_target = pad_sequences(train_target, padding='post')
val_input = pad_sequences(val_input, padding='post')
val_target = pad_sequences(val_target, padding='post')
test_input = pad_sequences(test_input, padding='post')
test_target = pad_sequences(test_target, padding='post')

print(get_bleu(test_input, test_target, source_tokenizer, target_tokenizer, loaded_model))