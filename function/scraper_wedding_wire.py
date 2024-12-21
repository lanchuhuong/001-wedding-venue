import base64
import hashlib
import io
import json
import os
import time
import urllib.parse

import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

load_dotenv(override=True)


class SeleniumDownloader:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920x1080")
        # Add user agent
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )

        self.driver = webdriver.Chrome(options=chrome_options)

    def get_image(self, url):
        """Download image using Selenium"""
        try:
            # First visit WeddingWire to set up cookies
            self.driver.get("https://www.weddingwire.com")
            time.sleep(2)  # Wait for cookies to be set

            # Now get the image
            self.driver.get(url)
            time.sleep(2)  # Wait for image to load

            # Get the image as base64
            img_base64 = self.driver.execute_script("""
                var c = document.createElement('canvas');
                var ctx = c.getContext('2d');
                var img = document.querySelector('img');
                
                if (!img) return null;
                
                c.height = img.naturalHeight;
                c.width = img.naturalWidth;
                ctx.drawImage(img, 0, 0);
                
                return c.toDataURL('image/jpeg').split(',')[1];
            """)

            if img_base64:
                return base64.b64decode(img_base64)
            return None

        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            return None

    def __del__(self):
        """Clean up the browser when done"""
        try:
            self.driver.quit()
        except:
            pass


class ChangeTracker:
    def __init__(self, bucket, tracker_path="image_tracker.json"):
        self.bucket = bucket
        self.tracker_path = tracker_path
        self.tracked_images = self.load_tracker()

    def load_tracker(self):
        try:
            blob = self.bucket.blob(self.tracker_path)
            content = blob.download_as_string()
            return json.loads(content)
        except Exception:
            return {}

    def save_tracker(self):
        blob = self.bucket.blob(self.tracker_path)
        blob.upload_from_string(json.dumps(self.tracked_images, indent=2))

    def get_url_hash(self, url):
        return hashlib.md5(url.encode()).hexdigest()

    def should_download(self, url, venue_name):
        url_hash = self.get_url_hash(url)
        venue_data = self.tracked_images.get(venue_name, {})
        return url_hash not in venue_data

    def mark_downloaded(self, url, venue_name, filename):
        url_hash = self.get_url_hash(url)
        if venue_name not in self.tracked_images:
            self.tracked_images[venue_name] = {}
        self.tracked_images[venue_name][url_hash] = {
            "filename": filename,
            "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "url": url,
        }


def get_filename_from_url(url, photo_col):
    parsed_url = urllib.parse.urlparse(url)
    path = parsed_url.path
    original_filename = os.path.basename(path)

    if original_filename and len(original_filename) > 10:
        return f"extra_{original_filename}"
    else:
        extension = os.path.splitext(parsed_url.path)[1] or ".jpg"
        return f"extra_{photo_col}{extension}"


def process_venues_and_photos(bucket_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    downloader = SeleniumDownloader()
    tracker = ChangeTracker(bucket)

    # Read Excel file
    excel_blob = bucket.blob("Wedding Venues.xlsx")
    excel_content = excel_blob.download_as_bytes()
    df = pd.read_excel(io.BytesIO(excel_content))

    photo_columns = [col for col in df.columns if "photo" in col.lower()]
    results = {"success": 0, "failed": 0, "skipped": 0}

    try:
        for index, row in df.iterrows():
            venue_name = str(row["Venue name"])
            print(f"\nProcessing venue: {venue_name}")

            for photo_col in photo_columns:
                url = row[photo_col]

                if pd.isna(url) or not url:
                    continue

                if not tracker.should_download(url, venue_name):
                    print(
                        f"Skipping already downloaded image for {venue_name}: {photo_col}"
                    )
                    results["skipped"] += 1
                    continue

                print(f"Downloading new image {photo_col}...")

                image_content = downloader.get_image(url)
                if image_content:
                    filename = get_filename_from_url(url, photo_col)
                    gcs_path = (
                        f"processed/adobe_extracted/{venue_name}/figures/{filename}"
                    )

                    try:
                        blob = bucket.blob(gcs_path)
                        blob.upload_from_string(
                            image_content, content_type="image/jpeg"
                        )
                        tracker.mark_downloaded(url, venue_name, filename)
                        print(f"✓ Successfully uploaded {filename}")
                        results["success"] += 1
                    except Exception as e:
                        print(f"✗ Failed to upload {filename}: {str(e)}")
                        results["failed"] += 1
                else:
                    print(f"✗ Failed to download {photo_col}")
                    results["failed"] += 1

        # Save the updated tracker
        tracker.save_tracker()

    finally:
        # Make sure we clean up the browser
        del downloader

    return results
