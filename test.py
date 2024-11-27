# import pandas as pd
# # df_cm = pd.read_csv(r'data/preprocessed_all.cm', delimiter='\t', header=None, names=['ID', 'Command'])
# # df_nl = pd.read_csv('data/preprocessed_all.nl', delimiter='\t', header=None, names=['ID', 'Description'])
# #
# # print(df_cm.tail())
# # print(df_nl.tail())
# df_cm = pd.read_csv('data/all.cm', delimiter='\t', header=None, names=['ID', 'Command'])
# df_nl = pd.read_csv('data/all.nl', delimiter='\t', header=None, names=['ID', 'Description'])
#
# # 检查数据加载情况
# print(f"Number of rows in Commands dataset: {len(df_cm)}")
# print(f"Number of rows in Descriptions dataset: {len(df_nl)}")
# print(df_nl.head())
# print(df_cm.head())
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from XiaoFishBot import Transformer, CustomTransformer, CustomSchedule
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import datetime
import os
import re
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=8, help='number of layers in transformer')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--dff', type=int, default=512, help='dimension of feed forward network')
    parser.add_argument('--num_heads', type=int, default=8, help='number of heads in multi-head attention')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--input_vocab_size', type=int, default=None, help='input vocabulary size')
    parser.add_argument('--target_vocab_size', type=int, default=None, help='target vocabulary size')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    param = parser.parse_args()
    return param


class DataLoader:
    def __init__(self, input_vocab_size, target_vocab_size, batch_size):
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.batch_size = batch_size


param = parse_args()
input_vocab_size = param.input_vocab_size
target_vocab_size = param.target_vocab_size
BATCH_SIZE = param.batch_size
num_layers = param.num_layers
d_model = param.d_model
dff = param.dff
num_heads = param.num_heads
dropout_rate = param.dropout_rate
epochs = param.epochs

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

source_tokenizer = Tokenizer(num_words=input_vocab_size, oov_token="<OOV>")
target_tokenizer = Tokenizer(num_words=target_vocab_size, oov_token="<OOV>", filters='')

# 训练分词器
source_tokenizer.fit_on_texts(source_sentences)
target_tokenizer.fit_on_texts(target_sentences)

# 将句子转换为序列
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

# 填充序列
train_input = pad_sequences(train_input, padding='post')
train_target = pad_sequences(train_target, padding='post')
val_input = pad_sequences(val_input, padding='post')
val_target = pad_sequences(val_target, padding='post')
test_input = pad_sequences(test_input, padding='post')
test_target = pad_sequences(test_target, padding='post')


# 创建 tar_inp 和 tar_real
def create_tar_inp_tar_real(tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    return tar_inp, tar_real


# 对训练集
train_tar_inp, train_tar_real = create_tar_inp_tar_real(train_target)
train_dataset = tf.data.Dataset.from_tensor_slices(((train_input, train_tar_inp), train_tar_real))
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
    BATCH_SIZE,
    padded_shapes=(([None], [None]), [None]),
    drop_remainder=True
)

# 对验证集
val_tar_inp, val_tar_real = create_tar_inp_tar_real(val_target)
val_dataset = tf.data.Dataset.from_tensor_slices(((val_input, val_tar_inp), val_tar_real))
val_dataset = val_dataset.padded_batch(
    BATCH_SIZE,
    padded_shapes=(([None], [None]), [None]),
    drop_remainder=True
)

# 对测试集
test_tar_inp, test_tar_real = create_tar_inp_tar_real(test_target)
print(test_tar_real[0])
test_dataset = tf.data.Dataset.from_tensor_slices(((test_input, test_tar_inp), test_tar_real))
test_dataset = test_dataset.padded_batch(
    BATCH_SIZE,
    padded_shapes=(([None], [None]), [None]),
    drop_remainder=True
)

example = next(iter(train_dataset))  # 获取一个 batch 的数据
source_texts, t_lb1 = example[0]  # 解构输入 (source_texts, tar_inp)
true_label = example[1]  # 解构标签 (tar_real)


def sequences_to_texts(sequences, tokenizer):
    return [
        " ".join([tokenizer.index_word.get(idx, "<OOV>") for idx in sequence if idx > 0])
        for sequence in sequences
    ]


def sequence_to_text(sequence, tokenizer):
    return " ".join([tokenizer.index_word.get(idx, "<OOV>") for idx in sequence if idx > 0])


# 转换预测序列为文字
print(target_tokenizer.index_word)
# 输出前几条预测结果
for i, (t_label, tlb) in enumerate(zip(target_sequences[0:5], true_label[0:5])):
    print(f"tar_inp translation：{sequence_to_text(tlb.numpy(), target_tokenizer)}")
    print(f"tar_inp translation：{tlb.numpy()}")
    print(f"Real translation：{sequence_to_text(t_label, target_tokenizer)}")
    print(f"Real translation：{t_label}")
