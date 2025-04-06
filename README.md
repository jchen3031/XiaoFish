# 🐟 XiaoFish

XiaoFish is a TensorFlow-based project for sequence modeling. Below is a guide to help you set up and run the project smoothly.

------

## 🚀 Installation Guide

To reproduce our experiments, **TensorFlow 2.0** is required. The experiments can be run on both GPU and CPU, but:

- **Training on CPU only** is extremely slow and **not recommended**.
- **Inference on CPU** is slow but **tolerable**.

We recommend using a **virtual environment** to avoid conflicts with other libraries.

### ✅ Recommended System Configuration

- **OS**: Ubuntu 16.04
- **CUDA**: 10.0
- **cuDNN**: 7.6.3

### 🔧 Installation Steps (for Windows & Linux)

```bash
conda create --name xiaofish python=3.9 -y
conda activate xiaofish
pip install tensorflow==2.10
pip install -r requirements.txt
```

After this, your environment is ready to use!

------

## 🏋️‍♀️ Train the Model

To train the model on the training set:

```bash
python main.py
```

Use `--epochs` to control training duration, for example:

```bash
python main.py --epochs 5
```

> 💡 We recommend training for at least **10 epochs** for better performance.

For a full list of hyperparameters:

```bash
python main.py --help
```

------

## 📈 Monitor Training with TensorBoard

To visualize training logs, run:

```bash
tensorboard --logdir=logs
```

Then open the displayed link in your browser.

------

## 🧠 Load & Evaluate the Pretrained Model

To load the pretrained model and test it:

```bash
python load_model.py
```

To generate prediction for a custom sentence:

```bash
python load_model.py --input_sentence "your_sentence"
```

To calculate BLEU score:

```bash
python load_model.py --bleu True
```

To see all options:

```bash
python load_model.py --help
```

> 📁 The best model is saved in the `latest_model/` folder.
>  To use it, either:
>
> - Keep the default `model_path`, or
>
> - Specify a custom model path:
>
>   ```bash
>   python load_model.py --model_path your_model_path
>   ```

------

## 🧪 Run Final Evaluation

To test the model using both BLEU and custom test method:

```bash
python test.py
```

> ⚠️ Note: `test.py` is **Linux-specific**, as it uses shell commands that may not work on Windows.

------

