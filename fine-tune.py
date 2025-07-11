import os
import openai
from openai import OpenAI
from load_dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
client = OpenAI(api_key=API_KEY)

ID = None

# Upload the training file
def train_file():
    train_file = openai.files.create(
        file=open("alchemy_finetune.jsonl", "rb"),
        purpose="fine-tune"
    )
    print("Training file ID:", train_file.id)


    response = openai.fine_tuning.jobs.create(
        training_file=train_file.id,
        model="gpt-4o-mini-2024-07-18",  # Use "gpt-4.1-mini" if available in your region/account
        hyperparameters={
            "n_epochs": 5,  # or more for better learning
            "batch_size": 8,
            "learning_rate_multiplier": 0.2
        }
    )
    print("Fine-tuning job ID:", response.id)
    return response.id


if __name__ == "__main__":
    
    ID = train_file()
    status = openai.fine_tuning.jobs.retrieve(ID)
    print("Status:", status.status)
    """
    completion = openai.chat.completions.create(
    model="ft:gpt-4o-mini-2024-07-18:your-org::xxxx",       # Change this to the officially returned fine-tuned model
    messages=[
        {"role": "user", "content": "Is 'Armchair' an official element in Little Alchemy 2?"}
    ])
    print(completion.choices[0].message.content)
    """

