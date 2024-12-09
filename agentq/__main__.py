import asyncio
from playwright.async_api import Page
import sounddevice as sd
from scipy.io.wavfile import write
import openai
from dotenv import load_dotenv
import os
import requests
import pyttsx3  # Text-to-speech library

from agentq.core.agent.agentq import AgentQ
from agentq.core.agent.agentq_actor import AgentQActor
from agentq.core.agent.agentq_critic import AgentQCritic
from agentq.core.agent.browser_nav_agent import BrowserNavAgent
from agentq.core.agent.planner_agent import PlannerAgent
from agentq.core.models.models import State
from agentq.core.orchestrator.orchestrator import Orchestrator

from instructor import from_openai, Mode
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('voice', 'com.apple.speech.synthesis.voice.daniel')

# Map states to agents
state_to_agent_map = {
    State.PLAN: PlannerAgent(),
    State.BROWSE: BrowserNavAgent(),
    State.AGENTQ_BASE: AgentQ(),
    State.AGENTQ_ACTOR: AgentQActor(),
    State.AGENTQ_CRITIC: AgentQCritic(),
}

# Function to record audio
def record_audio(filename="input.wav", duration=10, fs=44100):
    print("Recording... Speak into the microphone.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print("Recording complete.")

# Function to transcribe audio using Whisper API
def transcribe_audio(filename="input.wav"):
    headers = {
        "Authorization": f"Bearer {openai.api_key}"
    }
    files = {
        "file": (filename, open(filename, "rb")),
        "model": (None, "whisper-1")
    }
    response = requests.post("https://api.openai.com/v1/audio/transcriptions", headers=headers, files=files)
    response_data = response.json()
    print("[DEBUG] Transcription API Response:", response_data)
    return response_data.get("text", "Transcription failed.")

# Define the response model
class EnhancedPromptResponse(BaseModel):
    content: str

# Initialize instructor client
client = from_openai(openai.Client(), mode=Mode.JSON)

# Enhance the user's input prompt for clarity and alignment with Agent-Q's capabilities
def enhance_prompt(user_input):
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a command refiner for a web automation agent. Expand the user's input into a clear, detailed, and actionable command."
            },
            {
                "role": "user",
                "content": f"Expand the following command so that it is clear and actionable: {user_input}",
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=messages,
            response_model=EnhancedPromptResponse,
        )

        enhanced_prompt = response.content.strip()
        print(f"[DEBUG] Enhanced Prompt: {enhanced_prompt}")
        return enhanced_prompt
    except Exception as e:
        print(f"[ERROR] Error enhancing prompt: {e}")
        return user_input

# Helper to get command input via typing or speech
def get_command_input():
    choice = input("Press 's' to speak or 't' to type your command: ").lower()
    if choice == "s":
        record_audio()
        command = transcribe_audio()
        print(f"You said: {command}")
    else:
        command = input("Enter your command: ")

    enhanced_command = enhance_prompt(command)
    print(f"[INFO] Final Enhanced Command: {enhanced_command}")
    return enhanced_command

async def run_agent(command):
    # Speak a static confirmation response
    confirmation = "Absolutely, let me get that started for you."
    print(f"[INFO] Speaking Response: {confirmation}")
    engine.say(confirmation)
    engine.runAndWait()

    # Execute the task
    print(f"[INFO] Command passed to Agent: {command}")
    orchestrator = Orchestrator(state_to_agent_map=state_to_agent_map, eval_mode=True)
    await orchestrator.start()

    page: Page = await orchestrator.playwright_manager.get_current_page()
    await page.set_extra_http_headers({"User-Agent": "AgentQ-Bot"})

    print("[INFO] Executing command...")
    result = await orchestrator.execute_command(command)
    print("[INFO] Task Execution Complete.")
    return result

# Main loop
async def main():
    while True:
        command = get_command_input()
        if command.lower() == "exit":
            print("Exiting Agent Q.")
            break
        result = await run_agent(command)
        print("Result:", result)

if __name__ == "__main__":
    asyncio.run(main())
