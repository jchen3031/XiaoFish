import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from XiaoFishBot import Transformer, CustomTransformer, CustomSchedule
from utils import predict_sequence_step_by_step
import numpy as np
# 定义保存的路径load
model_dir = "latest_model"
tokenizer_path = os.path.join(model_dir, "tokenizers")
saved_model_path = os.path.join(model_dir, "saved_model")
SEED = 42
np.random.seed(SEED)
# 加载模型
loaded_model = load_model(saved_model_path, custom_objects={
    "Transformer": Transformer,
    "CustomTransformer": CustomTransformer,
    "CustomSchedule": CustomSchedule
})
# 计算模型的总参数数量
total_params = loaded_model.count_params()
print(f"Total parameters: {total_params}")
# 累加所有权重的参数数量
total_params = sum(np.prod(var.shape) for var in loaded_model.trainable_variables)
print(f"Total trainable parameters: {total_params}")