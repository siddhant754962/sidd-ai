from huggingface_hub import whoami
from config import HF_TOKEN

try:
    user = whoami(token=HF_TOKEN)
    print("Hugging Face authentication successful!")
    print("Username:", user["name"])

except Exception as e:
    print("Hugging Face authentication failed.")
    print(type(e).__name__)
    print(e)