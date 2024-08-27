import logging
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw
import requests
import torch

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def main():
    model_name = "facebook/detr-resnet-50"
    logging.info(f"Loading model and processor for {model_name}")
    model = DetrForObjectDetection.from_pretrained(model_name)
    processor = DetrImageProcessor.from_pretrained(model_name)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    logging.info(f"Fetching image from {url}")
    image = Image.open(requests.get(url, stream=True).raw)

    logging.info("Preprocessing the image")
    inputs = processor(images=image, return_tensors="pt")

    logging.info("Performing inference")
    with torch.no_grad():
        outputs = model(**inputs)

    logging.info("Processing outputs")
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    logging.info("Detected objects:")
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        logging.info(f"Detected {model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at location {box}")

if __name__ == "__main__":
    main()
