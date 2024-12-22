import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from function.scraper_wedding_wire import process_venues_and_photos


def main():
    bucket_name = "wedding-venues-001"

    try:
        print("Starting image download and upload process...")
        results = process_venues_and_photos(bucket_name)
        print("\nProcessing completed!")
        print(f"Successfully processed: {results['success']} images")
        print(f"Skipped (already downloaded): {results['skipped']} images")
        print(f"Failed to process: {results['failed']} images")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
