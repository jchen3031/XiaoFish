import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from XiaoFishBot import Transformer, CustomTransformer, CustomSchedule
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import datetime
import os


print("GPU Available:", tf.config.list_physical_devices('GPU'))
# 读取文件内容
with open(r'data/all.nl', 'r', encoding='utf-8') as f_cm:
    source_sentences = f_cm.readlines()

with open(r'data/all.cm', 'r', encoding='utf-8') as f_nl:
    target_sentences = f_nl.readlines()

# source_sentences, target_sentences = source_sentences[:2000], target_sentences[:2000]
model_dir = "saved_models"
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
# 创建源和目标语言的 Tokenizer
# target_sentences = target_sentences[:100]
input_vocab_size = 8500
target_vocab_size = 8000
source_tokenizer = Tokenizer(num_words=input_vocab_size, oov_token="<OOV>")
target_tokenizer = Tokenizer(num_words=target_vocab_size, oov_token="<OOV>")

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
d_model = 128
dff = 256
num_heads = 16
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
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
# 编译模型
custom_transformer.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 训练模型
history = custom_transformer.fit(
    train_dataset,
    epochs=10,
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
# test_loss, test_accuracy = custom_transformer.evaluate(test_dataset)
# print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"))
pred = custom_transformer.predict(train_dataset.take(1))
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
