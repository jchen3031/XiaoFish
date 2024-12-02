import tensorflow as tf
import keras

print("TensorFlow Version:", tf.__version__)
print("Keras Version:", keras.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
