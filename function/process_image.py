import uuid
from mimetypes import guess_type
from typing import Dict, List, Optional
import base64
import json
import os
from openai import OpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


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


def generate_image_descriptions(
    base_dir: str,
    pdf_name: str,
    output_file: str = "description.json",
    model: str = "gpt-4o",
) -> List[Dict[str, str]]:
    """
    Generate descriptions for images in a directory using OpenAI's API.
    """

    client = OpenAI(api_key=openai_api_key)
    image_description = []

    # Create full path to figures directory
    path = os.path.join(base_dir, pdf_name)

    if not os.path.isdir(path):
        raise OSError(f"Directory not found: {path}")

    # Process each image in the directory
    for image_file in os.listdir(path):
        image_path = os.path.join(path, image_file)

        try:
            # Convert image to data URL
            data_url = local_image_to_data_url(image_path)

            # Generate description using OpenAI API
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """You are tasked with summarizing the description of the images. 
                                Give a concise summary of the images provided to you. Pay attention to the theme. The output should not be more than 30 words. """,
                            },
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                max_tokens=30,
            )

            content = response.choices[0].message.content

            image_description.append({"image_path": image_path, "description": content})

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue

    # Save descriptions to JSON file
    try:
        with open(output_file, "w") as file:
            json.dump(image_description, file, indent=4)
    except Exception as e:
        print(f"Error saving descriptions to file: {e}")

    return image_description
