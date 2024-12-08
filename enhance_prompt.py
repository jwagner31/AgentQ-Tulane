import openai
from instructor import from_openai, Mode
from dotenv import load_dotenv
import os
from pydantic import BaseModel

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the response model
class EnhancedPromptResponse(BaseModel):
    content: str

# Initialize instructor client
client = from_openai(openai.Client(), mode=Mode.JSON)

def enhance_prompt(user_input):
    """
    Enhances the user's input prompt for better clarity and alignment with Agent-Q's capabilities.
    """
    try:
        # Define the system and user messages
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a command refiner for a web automation agent. Expand the user's input into a clear, detailed, and actionable command."
                ),
            },
            {
                "role": "user",
                "content": f"Expand the following command so that it is clear and actionable: {user_input}",
            },
        ]

        # Call the instructor-wrapped OpenAI API with response model
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",  # Ensure you have access to GPT-4
            messages=messages,
            response_model=EnhancedPromptResponse,  # Specify the response model
        )

        # Extract the enhanced prompt
        enhanced_prompt = response.content.strip()
        print(f"[DEBUG] Enhanced Prompt: {enhanced_prompt}")
        return enhanced_prompt
    except Exception as e:
        print(f"[ERROR] Error enhancing prompt: {e}")
        return user_input  # Fallback to original input

if __name__ == "__main__":
    # Test the enhance_prompt function
    print("Testing enhance_prompt...")
    user_input = "Walmart eggs"
    enhanced_prompt = enhance_prompt(user_input)
    print(f"Final Enhanced Prompt: {enhanced_prompt}")
