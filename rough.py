# import logging.config
# import json

# with open('E:/OCR/aadhar_ocr/config/logger.json', 'r') as f:
#     config = json.load(f)
# logging.config.dictConfig(config)

# logger = logging.getLogger('error_log')
# print("Logger name:", logger.name)
# print("Logger effective level:", logging.getLevelName(logger.getEffectiveLevel()))
# for handler in logger.handlers:
#     print("Handler:", handler)
#     print("    Level:", logging.getLevelName(handler.level))
#     print("    Formatter:", handler.formatter._fmt if handler.formatter else None)
#     print("    Handler type:", type(handler))

# import requests

# url = "http://localhost:8000/process-file"
# files = {"file": ("aadhaar.jpg", open("E:/Chrome Downloads/Hiren_vaghela_documents/Hiren_Vaghela_photo_id.jpeg", "rb"), "image/jpeg")}
# data = {"ref_id": "TEST001"}

# response = requests.post(url, files=files, data=data)
# print(response.json())\
from pathlib import Path
print(Path(__file__).parent.resolve())