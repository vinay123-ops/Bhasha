import requests

url = "http://127.0.0.1:8001/translate-and-transliterate"

payload = {
    "sentences": ["This is a test sentence."],
    "target_script": "IAST",
    "src_lang": "eng_Latn",
    "tgt_lang": "hin_Deva"
}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Raw Response:")
print(response.text)  # Add this to see what's going wrong
