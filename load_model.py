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