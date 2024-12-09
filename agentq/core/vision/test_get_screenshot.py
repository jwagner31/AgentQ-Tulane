import asyncio
from agentq.core.web_driver.playwright import PlaywrightManager
from agentq.utils.logger import logger
from agentq.core.vision.get_screenshot import get_screenshot
from agentq.core.vision.vision_model import process_screenshot_with_gpt4_vision


async def test_screenshot_with_vision():
    print("Test script execution started.")
    logger.info("Starting test for get_screenshot with vision integration...")

    try:
        # Initialize PlaywrightManager
        print("Initializing PlaywrightManager...")
        browser_manager = PlaywrightManager(browser_type="chromium", headless=True)
        await browser_manager.async_initialize(homepage="https://www.google.com")
        print("PlaywrightManager initialized.")

        # Get the current page
        page = await browser_manager.get_current_page()
        print(f"Current page retrieved: {page.url}")

        # Verify page content
        content = await page.content()
        print(f"Page content length: {len(content)}")

        # Call get_screenshot
        print("Calling get_screenshot...")
        result = await get_screenshot(page)

        # Save the Base64-encoded screenshot
        screenshot_bytes = result["screenshot"].split(",")[1]
        with open("debug_screenshot.png", "wb") as f:
            f.write(base64.b64decode(screenshot_bytes))
        print("Screenshot saved as debug_screenshot.png.")

        # Print metadata
        print("\nScreenshot Metadata:")
        print(result["metadata"])

        # Vision Model Test
        print("Processing screenshot with Vision Model...")
        vision_results = await process_screenshot_with_gpt4_vision(result)
        print("\nVision Model Results:")
        print(vision_results)

        # Stop Playwright
        await browser_manager.stop_playwright()
        print("Playwright stopped.")

    except Exception as e:
        print(f"Test failed with exception: {e}")

    print("Test completed.")


if __name__ == "__main__":
    try:
        asyncio.run(test_screenshot_with_vision())
    except Exception as e:
        print(f"Unhandled exception: {e}")
