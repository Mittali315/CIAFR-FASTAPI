import pytesseract
from PIL import Image

def extract_text_from_image(image_path: str) -> str:
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text.lower()

def validate_document(text: str) -> bool:
    """
    Simple validation logic
    """
    required_keywords = [
        "government",
        "identity",
        "name",
        "date",
    ]

    for word in required_keywords:
        if word not in text:
            return False

    return True
