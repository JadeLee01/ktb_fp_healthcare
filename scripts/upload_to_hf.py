from huggingface_hub import HfApi, login
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN") 
REPO_ID = os.getenv("HF_REPO_ID", "20-team-daeng-ddang-ai/dog-pose-estimation")
MODEL_PATH = "models/best.pt" # Path to your trained model (adjusted to likely location)
UPLOAD_NAME = "YOLO26s-pose.pt" # Name to save as in the repo

def upload_model():
    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable not set in .env or system.")
        return

    print(f"Logging in to Hugging Face...")
    login(token=HF_TOKEN)
    api = HfApi()

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found.")
        return

    print(f"Uploading {MODEL_PATH} to {REPO_ID} as {UPLOAD_NAME}...")
    
    try:
        api.upload_file(
            path_or_fileobj=MODEL_PATH,
            path_in_repo=UPLOAD_NAME,
            repo_id=REPO_ID,
            repo_type="model"
        )
        print("Upload successful!")
        print(f"Model URL: https://huggingface.co/{REPO_ID}/blob/main/{UPLOAD_NAME}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found. Train the model first.")
    else:
        upload_model()
