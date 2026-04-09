# test_api.py
import requests

with open("test_images/coco_img.jpg", "rb") as f:
    files = {"file": f}
    data = {"goal": "balanced"}
    r = requests.post("http://localhost:8000/api/segment", files=files, data=data)

print(f"Status: {r.status_code}")
print(f"Response keys: {r.json().keys()}")
print(f"Method: {r.json().get('method')}")
print(f"Confidence: {r.json().get('confidence')}")
print(f"Chars: {r.json().get('chars')}")
print(f"Mask length: {len(r.json().get('mask_b64', ''))}")
