# XiaoFish
Install TensorFlow
To reproduce our experiments, please install TensorFlow 2.0. The experiments can be reproduced on machines with or without GPUs. However, training with CPU only is extremely slow and we do not recommend it. Inference with CPU only is slow but tolerable.

We suggest following the official instructions to install the library. The code has been tested on Ubuntu 16.04 + CUDA 10.0 + CUDNN 7.6.3.

We recommend using a virtual environment to avoid conflicts with other libraries. 

Here are the steps to install TensorFlow 2.0 by window and linux system:
```
conda create --name xiaofish python=3.9 -y
conda activate xiaofish
pip install tensorflow==2.10
pip install -r requirements.txt
```

Then the environment is ready to use. 

Run ```python main.py``` to train the model on the training set. Using parse_args to set the hyperparameters.
You can run
```python main.py --epochs 5``` to train the model for 5 epochs.<br>
To have a better performance of model, we suggest to train at least 10 epochs.<br>
```python main.py --help``` to get the list of hyperparameters.<br>
To track the training process, you can use tensorboard by running ```tensorboard --logdir=logs``` in the terminal.

You can use run ```python load_model.py``` to load the pre-trained model. And test on the test set to get the bleu score.<br>
```python load_model.py --input_sentence your_sentence``` to get the predicted sentence.<br>
```python load_model.py --bleu True``` to get the bleu score.<br>
```python load_model.py --help``` to get the list of hyperparameters.
Our best model is saved in the ```latest_model``` folder.
To use it in load_model.py, you can set the ```model_path``` to ```latest_model``` or keep it as default.
if you want to use your own model, you can change the ```model_path``` to your own model path.
```python load_model.py --model_path your_model_path```