import os
from openai import OpenAI
from typing import Dict
from functools import lru_cache
from rich import print
from load_dotenv import load_dotenv
import time

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
client = OpenAI(api_key=API_KEY)

# In-memory stores
content: Dict[str, str] = {}  
agents: Dict[str, dict] = {} 

def format_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")

def load_instructions_from_file(file_path: str, **kwargs) -> str:
    with open(file_path, 'r',encoding='utf-8') as file:
        template = file.read()
    return template.format(**kwargs)

@lru_cache(maxsize=128)
def create_dynamic_agent(user_id: str, name: str) -> str:
    formatted_name = format_name(name)
    instructions = load_instructions_from_file(
        "instruction.txt"
    )

    print(f"[cyan]Creating agent for [bold]{formatted_name}[/bold]...[/cyan]")

    assistant = client.beta.assistants.create(
        name=formatted_name,
        instructions=instructions,
        tools=[{"type": "file_search"}, {"type": "code_interpreter"}],
        model="gpt-4.1-mini",
    )

    agents[user_id] = {
        "name": formatted_name,
        "instructions": instructions,
        "assistant_id": assistant.id
    }

    print(f"[green]Agent created with ID: {assistant.id}[/green]")
    return assistant.id

def create_thread(uid: str) -> None:
    if uid not in content:
        thread = client.beta.threads.create()
        content[uid] = thread.id
        print(f"[blue]Thread created for UID: {uid}[/blue]")

def add_user_message(uid: str, message: str) -> None:
    if uid not in content:
        raise ValueError("Thread not initialized. Call create_thread first.")
    client.beta.threads.messages.create(
        thread_id=content[uid],
        role="user",
        content=message
    )
    print(f"[yellow]User message added to thread {content[uid]}[/yellow]")

def get_response(uid: str) -> str:
    if uid not in agents:
        raise ValueError(f"No agent found for UID {uid}. Call create_dynamic_agent first.")

    run = client.beta.threads.runs.create(
        thread_id=content[uid],
        assistant_id=agents[uid]["assistant_id"]
    )

    print(f"[magenta]Waiting for response...[/magenta]")
    while run.status not in ["completed", "failed", "cancelled"]:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(
            thread_id=content[uid],
            run_id=run.id
        )

    if run.status != "completed":
        raise RuntimeError(f"Run did not complete successfully: {run.status}")

    messages = client.beta.threads.messages.list(thread_id=content[uid])
    for msg in reversed(messages.data):
        if msg.role == "assistant":
            return msg.content[0].text.value

    return "[Error: No assistant message found]"

def delete_agent(user_id: str) -> None:
    """Delete the assistant and clean up memory."""
    if user_id in agents:
        assistant_id = agents[user_id]["assistant_id"]
        client.beta.assistants.delete(assistant_id)
        agents.pop(user_id, None)
        create_dynamic_agent.cache_clear()
        print(f"[red]Deleted assistant with ID: {assistant_id}[/red]")
    else:
        print(f"[gray]No agent found for UID: {user_id} to delete.[/gray]")

if __name__ == "__main__":
    import json
    uid = "user_123"
    name = "Cool Assistant"

    create_thread(uid)
    create_dynamic_agent(uid, name)
    add_user_message(uid, 'Using the base elements: ["air", "water", "fire", "earth"], generate the deepest possible tree of realistic combinations, following the system rules. Output only the final JSON.')
    reply = get_response(uid)
    try:
        parsed_reply = json.loads(reply.replace("json","").replace("`", ""))
        with open("reply_v4.json", "w", encoding="utf-8") as f:
            json.dump(parsed_reply, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[red]Error parsing JSON: {e}[/red]")
        print("[red]Failed to parse assistant reply as JSON.[/red]")
        with open("reply_raw.txt", "w", encoding="utf-8") as f:
            f.write(reply)
    delete_agent(uid)
    print("[blue]Agent deleted.[/blue]")
    
