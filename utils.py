import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu

def predict_sequence_step_by_step(input_sentence, model, source_tokenizer, target_tokenizer, max_length=50,
                                  beam_width=5):
    # 1. 预处理输入句子
    input_sentence = preprocess_text(input_sentence)
    input_sequence = source_tokenizer.texts_to_sequences([input_sentence])
    input_sequence = pad_sequences(input_sequence, padding='post')

    # 2. 初始化解码器输入序列（以 <START> 开始）
    start_token = target_tokenizer.word_index.get('<start>', None)
    end_token = target_tokenizer.word_index.get('<end>', None)
    if start_token is None or end_token is None:
        raise ValueError("<START> or <END> token is missing in the target tokenizer.")

    target_sequence = [start_token]  # 初始目标序列

    # 3. 逐步解码
    for _ in range(max_length):
        # 填充当前目标序列
        tar_inp = pad_sequences([target_sequence], padding='post', maxlen=max_length)

        # 调用模型预测下一个 token 的概率分布
        predictions = model.predict((input_sequence, tar_inp), verbose=0)  # 输入包括源和当前目标序列
        predictions = tf.nn.softmax(predictions, axis=-1).numpy()

        # Beam Search 解码（或 Greedy Search 选择下一个 token）
        next_token = np.argmax(predictions[0, len(target_sequence) - 1])  # 当前时间步的预测
        target_sequence.append(next_token)  # 将预测的 token 添加到目标序列

        # 如果生成了结束标志 <END>，停止解码
        if next_token == end_token:
            break

    # 4. 将生成的序列转换为文本
    decoded_sequence = " ".join(
        [target_tokenizer.index_word.get(idx, "<OOV>") for idx in target_sequence if idx > 0]
    )
    return decoded_sequence

def sequence_to_text(sequence, tokenizer):
    return " ".join([tokenizer.index_word.get(idx, "<OOV>") for idx in sequence if idx > 0])

def preprocess_text(text):
    text = re.sub(r'[\t\n]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text.strip()

# BLEU Warpper
def get_bleu(test_input, test_target, source_tokenizer, target_tokenizer, custom_transformer):
    def compute_bleu_score(reference, prediction):
        smoothie = SmoothingFunction().method1
        return sentence_bleu([reference], prediction, smoothing_function=smoothie)
    # Evaluate BLEU on the test set
    print("----------------BLEU EVALUATION START----------------")
    bleu_scores = []
    for (source_text, target_text) in zip(test_input, test_target):
        # Convert source sequence to text
        source_sentence = sequence_to_text(source_text, source_tokenizer)
        # Predict the target sequence step by step
        predicted_sequence = predict_sequence_step_by_step(
            source_sentence,
            custom_transformer,
            source_tokenizer,
            target_tokenizer
        )
        prediction = predicted_sequence.split()
        reference = [target_tokenizer.index_word.get(idx, "<OOV>") for idx in target_text if idx > 0]
        # Compute BLEU score
        bleu = compute_bleu_score(reference, prediction)
        bleu_scores.append(bleu)
    average_bleu = np.mean(bleu_scores)
    print(f"Average BLEU score on test set: {average_bleu:.4f}")
    print("----------------BLEU EVALUATION END----------------")

