import base64
from typing_extensions import Annotated, Optional
from agentq.core.web_driver.playwright import PlaywrightManager
from agentq.utils.logger import logger
from playwright.async_api import Page
from agentq.core.vision.vision_model import process_screenshot_with_gpt4_vision

async def get_screenshot(
        webpage: Optional[Page] = None
) -> (
    Annotated[
        dict, "Returns a dictionary with base64 encoded screenshot, metadata, and vision insights."
    ]
):
    try:
        logger.info("get_screenshot: Starting process...")
        browser_manager = PlaywrightManager(browser_type="chromium", headless=False)

        if webpage is None:
            webpage = await browser_manager.get_current_page()
            logger.info("get_screenshot: Retrieved current page.")

        if not webpage:
            logger.error("get_screenshot: No active page found.")
            raise ValueError("No active page found. OpenURL command opens a new page.")

        # Ensure the page has loaded
        await webpage.wait_for_load_state("networkidle")
        logger.info("get_screenshot: Page load state confirmed.")

        # Capture the screenshot
        screenshot_bytes = await webpage.screenshot(full_page=False)
        logger.info("get_screenshot: Screenshot captured successfully.")

        # Save raw screenshot for debugging
        with open("debug_raw_screenshot.png", "wb") as f:
            f.write(screenshot_bytes)
        logger.info("get_screenshot: Raw screenshot saved as debug_raw_screenshot.png.")

        # Encode the screenshot as Base64
        base64_screenshot = base64.b64encode(screenshot_bytes).decode("utf-8")
        logger.info("get_screenshot: Screenshot encoded to Base64.")

        # Example metadata
        metadata = {
            "width": webpage.viewport_size["width"],
            "height": webpage.viewport_size["height"],
            "elements": []  # Placeholder for identified UI elements
        }

        # Package the screenshot data
        screenshot_data = {
            "screenshot": f"data:image/png;base64,{base64_screenshot}",
            "metadata": metadata
        }

        # Process the screenshot with the Vision Module
        vision_insights = await process_screenshot_with_gpt4_vision(screenshot_data)
        logger.info("get_screenshot: Vision processing completed.")

        # Add insights to the metadata
        screenshot_data["metadata"]["ui_elements"] = vision_insights["ui_elements"]

        return screenshot_data

    except Exception as e:
        logger.error(f"Error in get_screenshot: {e}")
        raise ValueError(
            "Failed to capture and process screenshot with vision. Ensure the page is open and accessible."
        ) from e
