from __future__ import annotations

import base64
import functools
import json
import os
import pickle
import shutil
import tempfile
from mimetypes import guess_type
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pandas as pd
import pytesseract

# import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from function.secrets import secrets

if TYPE_CHECKING:
    from sklearn.ensemble import RandomForestClassifier

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = current_dir / Path("model.bin")
client = OpenAI()


@functools.lru_cache
def load_is_photo_classifier() -> RandomForestClassifier:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


def image_properties(image_path) -> dict:
    with Image.open(image_path) as pil_image:
        width, height = pil_image.size
        total_image_area = width * height

        ocr_data = pytesseract.image_to_data(
            pil_image, output_type=pytesseract.Output.DATAFRAME
        )

    total_text_area = 0
    try:
        if not ocr_data.empty:
            valid_boxes = ocr_data[
                ocr_data["text"].notna() & (ocr_data["text"].str.strip() != "")
            ]
            total_text_area = sum(
                row["width"] * row["height"] for _, row in valid_boxes.iterrows()
            )
    except Exception:
        pass

    text_density = total_text_area / total_image_area if total_image_area > 0 else 0

    cv_image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    total_pixels = height * width

    edges = cv2.Canny(gray, 100, 200)
    edge_pixels = np.count_nonzero(edges)
    edge_density = edge_pixels / total_pixels

    color_std = np.std(cv_image, axis=(0, 1)).mean()

    return {
        "width": width,
        "height": height,
        "total_pixels": total_pixels,
        "text_density": text_density,
        "color_std": color_std,
        "edge_density": edge_density,
    }


def predict_image_quality(model, properties: dict) -> bool:
    X = pd.DataFrame(
        {
            "width": properties["width"],
            "height": properties["height"],
            "total_pixels": properties["total_pixels"],
            "text_density": properties["text_density"],
            "color_std": properties["color_std"],
            "edge_density": properties["edge_density"],
        },
        index=[0],
    )
    return bool(model.predict(X)[0])


# @st.cache_data
def is_photo(_model, image_path) -> bool:
    properties = image_properties(image_path)
    return predict_image_quality(_model, properties)


def resize_image(image_path, max_size=512):
    """
    Resize an image if either width or height exceeds max_size while maintaining aspect ratio.
    Saves the result to a temporary file only if resizing is needed.

    Args:
        image_path (str): Path to input image
        max_size (int): Maximum allowed dimension (width or height). Defaults to 500.

    Returns:
        str: Either the original path if no resize needed, or path to temp file if resized
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size

            if width <= max_size and height <= max_size:
                return image_path

            scale = min(max_size / width, max_size / height)

            new_width = int(width * scale)
            new_height = int(height * scale)

            file_ext = os.path.splitext(image_path)[1]
            if not file_ext:
                file_ext = ".png"

            temp_file = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
            temp_path = temp_file.name

            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_img.save(temp_path)

            return temp_path

    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")


def local_image_to_data_url(image_path: str) -> str:
    """
    Convert a local image file to a data URL.
    """
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{base64_encoded_data}"


# @st.cache_data
def generate_image_descriptions(
    base_dir: str,
    venue: str,
    # output_file: str = "description.json",
    model: str = "gpt-4o",
) -> list[dict[str, str]]:
    """
    Generate descriptions for images in a directory using OpenAI's API.
    """
    print("Generating image descriptions...")
    client = OpenAI(api_key=secrets.OPENAI_API_KEY.get_secret_value())
    image_description = []

    if not os.path.isdir(base_dir):
        raise OSError(f"Directory not found: {base_dir}")

    for i, image_file in enumerate(os.listdir(base_dir)):
        image_path = os.path.join(base_dir, image_file)
        print(f"   ({i+1}/{len(os.listdir(base_dir))}) {image_path}")
        photo_classifier = load_is_photo_classifier()
        if not is_photo(photo_classifier, image_path):
            print(f"skipping {image_path}")
            continue
        temp_image_path = resize_image(image_path)
        try:
            data_url = local_image_to_data_url(temp_image_path)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """
                                    You are tasked with summarizing the description of the images.
                                    Give a concise summary of the images provided to you. Pay attention to the
                                    theme. The output should not be more than 30 words. """,
                            },
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                max_tokens=30,
            )

            content = response.choices[0].message.content
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue

        output_image_dir = Path(os.getenv("OUTPUT_IMAGES_DIR")) / venue
        output_image_dir.mkdir(exist_ok=True)
        output_image_path = output_image_dir / image_file
        shutil.copy(image_path, output_image_path)
        image_description.append(
            {"image_path": str(output_image_path), "description": content}
        )

    return image_description
