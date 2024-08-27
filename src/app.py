import logging
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import requests
import torch
import random
from config import Config

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

config = Config()
logging.info(f"CONFIG {config}")

MODEL_NAME = config.get("model_name")
SAMPLE_IMAGE_URL = config.get("sample_image_url")
OUTPUT_IMAGE_PATH = config.get("output_image_path")


def main():
    logging.info(f"Loading model and processor for {MODEL_NAME}")
    model = DetrForObjectDetection.from_pretrained(MODEL_NAME)
    processor = DetrImageProcessor.from_pretrained(MODEL_NAME)

    logging.info(f"Fetching image from {SAMPLE_IMAGE_URL}")
    image = Image.open(requests.get(SAMPLE_IMAGE_URL, stream=True).raw)

    logging.info("Preprocessing the image")
    inputs = processor(images=image, return_tensors="pt")

    logging.info("Performing inference")
    with torch.no_grad():
        outputs = model(**inputs)

    logging.info("Processing outputs")
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    logging.info("Detected objects:")
    draw = ImageDraw.Draw(image)
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan", "magenta"]
    font = ImageFont.load_default()

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 2)}"
        color = random.choice(colors)  # Randomly choose a color for each box
        logging.info(f"label: {label},{type(label)}")
        logging.info(f"Detected {label_text} at location {box}")

        # Draw the bounding box on the image
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0], box[1]), label_text, fill="white", font=font)

    # Save the image with bounding boxes
    image.save(OUTPUT_IMAGE_PATH)
    logging.info(f"Saved output image with bounding boxes to {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    main()
