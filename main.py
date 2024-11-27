import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from XiaoFishBot import Transformer, CustomTransformer, CustomSchedule
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import datetime
import os
import argparse
import re
from utils import preprocess_text, sequence_to_text, predict_sequence_step_by_step
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=4, help='number of layers in transformer')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--dff', type=int, default=512, help='dimension of feed forward network')
    parser.add_argument('--num_heads', type=int, default=8, help='number of heads in multi-head attention')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--input_vocab_size', type=int, default=10000, help='input vocabulary size')
    parser.add_argument('--target_vocab_size', type=int, default=10000, help='target vocabulary size')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--model_dir', type=str, default='saved_models', help='saved_models directory')
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

filters = r'!"#$%&()*+,;=?@[\\]^_`{|}~'
filters = filters.replace('|', '')

source_sentences = [preprocess_text(s) for s in source_sentences]
target_sentences = [preprocess_text(s) for s in target_sentences]
target_sentences = [f"<START> {s} <END>" for s in target_sentences]
# source_sentences, target_sentences = source_sentences[:2000], target_sentences[:2000]
model_dir = param.model_dir
checkpoint_dir = "checkpoints"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "transformer_checkpoint_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5")
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,  # 只保存权重
    save_best_only=True,     # 只保存最好的模型
    monitor='val_loss',      # 监控验证集损失
    mode='min',              # 监控指标越小越好
    verbose=1
)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
source_tokenizer = Tokenizer(num_words=input_vocab_size, oov_token="<OOV>", filters='')
target_tokenizer = Tokenizer(num_words=target_vocab_size, oov_token="<OOV>", filters='')

# 训练分词器
source_tokenizer.fit_on_texts(source_sentences)
target_tokenizer.fit_on_texts(target_sentences)
# target_tokenizer.fit_on_texts(["<START>", "<END>"] + target_sentences)
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


# 创建 Transformer 模型
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    pe_input=200,
    pe_target=100,
    rate=dropout_rate
)

# 创建自定义模型
custom_transformer = CustomTransformer(transformer)
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
# 编译模型
custom_transformer.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 训练模型
if epochs > 0:
    history = custom_transformer.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[tensorboard_callback]
    )
saved_model_path = os.path.join(model_dir, "transformer_model")
custom_transformer.save(saved_model_path)
print(f"Model saved to {saved_model_path}")
import pickle

tokenizer_path = os.path.join(model_dir, "tokenizers")
os.makedirs(tokenizer_path, exist_ok=True)
# 保存源语言分词器
with open(os.path.join(tokenizer_path, 'source_tokenizer.pickle'), 'wb') as handle:
    pickle.dump(source_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 保存目标语言分词器
with open(os.path.join(tokenizer_path, 'target_tokenizer.pickle'), 'wb') as handle:
    pickle.dump(target_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Tokenizers saved successfully")
test_loss, test_accuracy = custom_transformer.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
example = next(iter(train_dataset))  # 获取一个 batch 的数据
source_texts, _ = example[0]  # 解构输入 (source_texts, tar_inp)
true_label = example[1]  # 解构标签 (tar_real)
pred = custom_transformer.predict(example[0])
pred = tf.nn.softmax(pred, axis=-1).numpy()
# print(pred[:5], pred.shape)
predicted_sequences = np.argmax(pred, axis=-1)


def sequences_to_texts(sequences, tokenizer):
    return [
        " ".join([tokenizer.index_word.get(idx, "<OOV>") for idx in sequence if idx > 0])
        for sequence in sequences
    ]


def beam_search_decoder(pred, beam_width, tokenizer):
    decoded_sequences = []
    for pred_sequence in pred:

        sequences = [[[], 0.0]]
        for prob_distribution in pred_sequence:
            all_candidates = []
            for seq, score in sequences:
                for idx, prob in enumerate(prob_distribution):
                    if prob > 1e-9:
                        candidate = (seq + [idx], score - np.log(prob))
                        all_candidates.append(candidate)

            sequences = sorted(all_candidates, key=lambda tup: tup[1])[:beam_width]

        best_sequence = sequences[0][0]
        decoded_sequence = " ".join(
            [tokenizer.index_word.get(idx, "<OOV>") for idx in best_sequence if idx > 0]
        )
        decoded_sequences.append(decoded_sequence)
    return decoded_sequences



beam_width = 5  # Beam Search 的宽度
predicted_texts = beam_search_decoder(pred[:5], beam_width, target_tokenizer)


# 转换预测序列为文字
# predicted_texts = sequences_to_texts(predicted_sequences, target_tokenizer)
# print(target_tokenizer.index_word)
# 输出前几条预测结果
for i, (source_text, predicted_text, predicted_sequence) in enumerate(zip(source_texts.numpy()[:5], predicted_texts[:5], predicted_sequences[:5])):
    print(f"Example: {source_text}")
    print(f"Example {i + 1}: {sequence_to_text(source_text, source_tokenizer)}")
    print(f"Predicted translation：{predicted_sequence}")
    print(f"Predicted translation：{predicted_text}")
    print(f"Real translation：{sequence_to_text(true_label[i].numpy(), target_tokenizer)}")
    print(f"Real translation：{true_label[i].numpy()}")


# def predict_sentence(input_sentence, model, source_tokenizer, target_tokenizer, beam_width=5):
#     # 1. 预处理输入句子
#     input_sentence = preprocess_text(input_sentence)
#     input_sequence = source_tokenizer.texts_to_sequences([input_sentence])
#     input_sequence = pad_sequences(input_sequence, padding='post')
#
#     # 2. 调用模型预测
#     pred = model.predict((input_sequence, input_sequence))  # 注意 `tar_inp` 可能需要调整
#     pred = tf.nn.softmax(pred, axis=-1).numpy()
#
#     # 3. 使用 Beam Search 解码
#     predicted_text = beam_search_decoder(pred, beam_width, target_tokenizer)[0]
#     return predicted_text


input_sentence = "An no-op on filename with sed"
predicted_translation = predict_sequence_step_by_step(input_sentence, custom_transformer, source_tokenizer, target_tokenizer)
print(f"Input: {input_sentence}")
print(f"Predicted Translation: {predicted_translation}")
