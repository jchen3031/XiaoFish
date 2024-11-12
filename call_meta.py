import subprocess

script_path_llama = r"D:\CondaEnv\llama-models\Scripts\python.exe"
model_location = r"C:\Users\jchen\OneDrive\桌面\Chen\llama-models"


def using_meta_model(message):
    result = subprocess.run([script_path_llama, f'{model_location}\\api.py', message],
                            stdout=subprocess.PIPE, text=True, encoding='utf-8', errors="ignore")
    return result.stdout.strip()