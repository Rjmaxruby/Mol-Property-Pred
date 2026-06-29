import os
import torch
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_multitask_gnn.pt",
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = None

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)