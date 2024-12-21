import sys

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

sys.path.append("..")

from function.scraper_wedding_wire import process_venues_and_photos

options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()), options=options
)


def main():
    bucket_name = "wedding-venues-001"

    print("Starting image download and upload process...")
    results = process_venues_and_photos(bucket_name)
    print("\nProcessing completed!")
    print(f"Successfully processed: {results['success']} images")
    print(f"Skipped (already downloaded): {results['skipped']} images")
    print(f"Failed to process: {results['failed']} images")


if __name__ == "__main__":
    main()
