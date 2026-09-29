import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

print("Hugging Face token loaded:", bool(HF_TOKEN))