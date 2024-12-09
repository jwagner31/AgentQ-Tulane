import base64
import io
from PIL import Image
import openai

async def process_screenshot_with_gpt4_vision(screenshot_data: dict) -> dict:
    """
    Processes the screenshot using GPT-4 Vision to identify UI elements and helpful insights.

    Args:
    - screenshot_data: dict, The dictionary containing base64-encoded screenshot and metadata.

    Returns:
    - dict: Insights and identified UI elements from the screenshot.
    """
    try:
        # Decode the Base64 screenshot
        screenshot_base64 = screenshot_data["screenshot"].split(",")[1]
        screenshot_bytes = base64.b64decode(screenshot_base64)
        image = Image.open(io.BytesIO(screenshot_bytes))
        logger.info("process_screenshot_with_gpt4_vision: Image successfully decoded.")

        # Debugging: Save the image for inspection
        image.save("vision_debug_screenshot.png", "PNG")
        logger.info("process_screenshot_with_gpt4_vision: Debug screenshot saved.")

        # Call the GPT-4 Vision API
        response = openai.ChatCompletion.create(
            model="gpt-4-vision",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant designed to analyze UI screenshots and extract actionable insights.",
                },
                {
                    "role": "user",
                    "content": "Analyze this screenshot to identify UI elements and other helpful details.",
                },
            ],
            files=[
                {
                    "name": "screenshot.png",
                    "type": "image/png",
                    "content": screenshot_bytes,
                }
            ],
        )
        logger.info("process_screenshot_with_gpt4_vision: API call successful.")

        # Parse GPT-4's response
        identified_elements = response["choices"][0]["message"]["content"]
        return {"ui_elements": identified_elements}

    except Exception as e:
        logger.error(f"process_screenshot_with_gpt4_vision failed: {e}")
        raise ValueError(f"Failed to process screenshot with GPT-4 Vision: {str(e)}")
