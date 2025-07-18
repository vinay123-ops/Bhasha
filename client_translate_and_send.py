import requests

url = "http://127.0.0.1:8000/translate"

payload = {
    "sentences": [
        "When I was young, I used to go to the park every day.",
        "We watched a new movie last week, which was very inspiring."
    ],
    "src_lang": "eng_Latn",
    "tgt_lang": "hin_Deva"
}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response JSON:")
print(response.json())
