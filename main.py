# from XiaoFishBot import Transformer, CustomTransformer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# # 读取文件内容
#
# with open(r'nl2bash/data/bash/all.cm', 'r', encoding='utf-8') as f_cm:
#     source_sentences = f_cm.readlines()
#
# with open(r'nl2bash/data/bash/all.nl', 'r', encoding='utf-8') as f_nl:
#     target_sentences = f_nl.readlines()
#
# print(source_sentences[:5], target_sentences[:5])
#
# from tensorflow.keras.preprocessing.text import Tokenizer
#
# # 创建源和目标语言的 Tokenizer
# source_tokenizer = Tokenizer(num_words=8500, oov_token="<OOV>")
# target_tokenizer = Tokenizer(num_words=8000, oov_token="<OOV>")
#
# # 训练分词器
# source_tokenizer.fit_on_texts(source_sentences)
# target_tokenizer.fit_on_texts(target_sentences)
#
# # 将句子转换为序列
# input_sequences = source_tokenizer.texts_to_sequences(source_sentences)
# target_sequences = target_tokenizer.texts_to_sequences(target_sentences)
# import tensorflow as tf
#
# BATCH_SIZE = 64
# BUFFER_SIZE = 20000
# data_size = len(input_sequences)
# train_size = int(0.8 * data_size)
# val_size = int(0.1 * data_size)
# test_size = data_size - train_size - val_size
#
# train_input = input_sequences[:train_size]
# train_target = target_sequences[:train_size]
#
# val_input = input_sequences[train_size:train_size + val_size]
# val_target = target_sequences[train_size:train_size + val_size]
#
# test_input = input_sequences[train_size + val_size:]
# test_target = target_sequences[train_size + val_size:]
#
# train_input = pad_sequences(train_input, padding='post')
# train_target = pad_sequences(train_target, padding='post')
# val_input = pad_sequences(val_input, padding='post')
# val_target = pad_sequences(val_target, padding='post')
# test_input = pad_sequences(test_input, padding='post')
# test_target = pad_sequences(test_target, padding='post')
# # 将数据转换为 tf.data.Dataset
# # 构建训练集
# train_dataset = tf.data.Dataset.from_tensor_slices((train_input, train_target))
# train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([None], [None]), drop_remainder=True)
#
# # 构建验证集
# val_dataset = tf.data.Dataset.from_tensor_slices((val_input, val_target))
# val_dataset = val_dataset.padded_batch(BATCH_SIZE, padded_shapes=([None], [None]), drop_remainder=True)
#
# # 构建测试集
# test_dataset = tf.data.Dataset.from_tensor_slices((test_input, test_target))
# test_dataset = test_dataset.padded_batch(BATCH_SIZE, padded_shapes=([None], [None]), drop_remainder=True)
#
#
# # sample_transformer = Transformer(
# #     num_layers=2, d_model=512, num_heads=8, dff=2048,
# #     input_vocab_size=8500, target_vocab_size=8000,
# #     pe_input=10000, pe_target=6000)
# #
# # custom_transformer = CustomTransformer(sample_transformer)
# #
# # custom_transformer.compile(
# #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
# #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
# #     metrics=['accuracy'])
# #
# # EPOCHS = 20  # 设置您想要的训练轮数
# #
# # history = custom_transformer.fit(
# #     train_dataset,
# #     epochs=EPOCHS,
# #     validation_data=val_dataset
# # )
# num_layers = 4
# d_model = 128
# dff = 512
# num_heads = 8
#
# input_vocab_size = 8500
# target_vocab_size = 8000
# dropout_rate = 0.1
#
# # 创建 Transformer 模型
# transformer = Transformer(
#     num_layers=num_layers,
#     d_model=d_model,
#     num_heads=num_heads,
#     dff=dff,
#     input_vocab_size=input_vocab_size,
#     target_vocab_size=target_vocab_size,
#     pe_input=1000,
#     pe_target=1000,
#     rate=dropout_rate
# )
#
# # 创建自定义模型
# custom_transformer = CustomTransformer(transformer)
#
# # 编译模型
# custom_transformer.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=['accuracy']
# )
#
# # 准备示例数据
# batch_size = 64
# sequence_length = 40  # 序列长度
# num_samples = 1000  # 样本数量
# history = custom_transformer.fit(
#     train_dataset,
#     epochs=10,
#     validation_data=val_dataset
# )
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from XiaoFishBot import Transformer, CustomTransformer
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# 读取文件内容
with open(r'data/all.nl', 'r', encoding='utf-8') as f_cm:
    source_sentences = f_cm.readlines()

with open(r'data/all.cm', 'r', encoding='utf-8') as f_nl:
    target_sentences = f_nl.readlines()

# 创建源和目标语言的 Tokenizer
# target_sentences = target_sentences[:100]

source_tokenizer = Tokenizer(num_words=8500, oov_token="<OOV>")
target_tokenizer = Tokenizer(num_words=8000, oov_token="<OOV>")

# 训练分词器
source_tokenizer.fit_on_texts(source_sentences)
target_tokenizer.fit_on_texts(target_sentences)

# 将句子转换为序列
input_sequences = source_tokenizer.texts_to_sequences(source_sentences)
target_sequences = target_tokenizer.texts_to_sequences(target_sentences)

BATCH_SIZE = 64
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
test_dataset = tf.data.Dataset.from_tensor_slices(((test_input, test_tar_inp), test_tar_real))
test_dataset = test_dataset.padded_batch(
    BATCH_SIZE,
    padded_shapes=(([None], [None]), [None]),
    drop_remainder=True
)

# 设置超参数
num_layers = 8
d_model = 256
dff = 512
num_heads = 16

input_vocab_size = 8500
target_vocab_size = 8000
dropout_rate = 0.1

# 创建 Transformer 模型
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    pe_input=1000,
    pe_target=1000,
    rate=dropout_rate
)

# 创建自定义模型
custom_transformer = CustomTransformer(transformer)

# 编译模型
custom_transformer.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 训练模型
history = custom_transformer.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset
)
# test_loss, test_accuracy = custom_transformer.evaluate(test_dataset)
# print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
pred = custom_transformer.predict(val_dataset)
pred = tf.nn.softmax(pred, axis=-1).numpy()
print(pred[:5], pred.shape)
predicted_sequences = np.argmax(pred, axis=-1)


def sequences_to_texts(sequences, tokenizer):
    return [
        " ".join([tokenizer.index_word.get(idx, "<OOV>") for idx in sequence if idx > 0])
        for sequence in sequences
    ]

def beam_search_decoder(pred, beam_width, tokenizer):
    """
    使用 Beam Search 进行解码
    """
    decoded_sequences = []
    for pred_sequence in pred:
        # 以 beam_width 为宽度选择候选路径
        sequences = [[[], 0.0]]  # 初始序列和得分
        for prob_distribution in pred_sequence:
            all_candidates = []
            for seq, score in sequences:
                for idx, prob in enumerate(prob_distribution):
                    if prob > 0:  # 只处理有效概率
                        candidate = (seq + [idx], score - np.log(prob + 1e-9))
                        all_candidates.append(candidate)
            # 按得分排序，选择前 beam_width 个候选序列
            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            sequences = ordered[:beam_width]
        # 最佳候选序列
        best_sequence = sequences[0][0]
        decoded_sequences.append(
            " ".join([tokenizer.index_word.get(idx, "<OOV>") for idx in best_sequence if idx > 0])
        )
    return decoded_sequences


# 使用 Beam Search 解码
beam_width = 10  # Beam Search 的宽度
# predicted_texts = beam_search_decoder(pred[:5], beam_width, target_tokenizer)


# 转换预测序列为文字
predicted_texts = sequences_to_texts(predicted_sequences, target_tokenizer)

# 输出前几条预测结果
for i, text in enumerate(predicted_texts[:5]):
    print(f"预测第 {i + 1} 条：{text}")
