from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import time
import sys
import psutil
import os

from indic_transliteration.sanscript import transliterate, DEVANAGARI, IAST  # ✅ Add script constants here

app = FastAPI(title="BhashaSetu Transliteration Service", version="1.0")

TRANSLATION_API_URL = "http://localhost:8000/translate"  # Translation service must be running

# Supported scripts map
target_script_map = {
    "IAST": IAST,
    # Add other script constants here if needed
}

class TransliterationRequest(BaseModel):
    sentences: List[str]
    target_script: str = "IAST"
    src_lang: str = "eng_Latn"
    tgt_lang: str = "hin_Deva"

class TransliterationResponse(BaseModel):
    original: str
    translated: str
    transliterated: str
    latency_ms: float
    translation_cost: float
    space_complexity_bytes: int
    time_complexity_ops: int

@app.get("/")
@app.get("/health")
def health_check():
    return {"status": "Transliteration service is running."}

@app.post("/translate-and-transliterate", response_model=List[TransliterationResponse])
def translate_and_transliterate(req: TransliterationRequest):
    print("[INFO] Received request for translation and transliteration.")
    print(f"[DEBUG] Input sentences: {req.sentences}")
    print(f"[DEBUG] Source language: {req.src_lang}")
    print(f"[DEBUG] Target language: {req.tgt_lang}")
    print(f"[DEBUG] Target script: {req.target_script}")

    start_time = time.perf_counter()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss

    try:
        print("[INFO] Sending request to translation service...")
        translation_payload = {
            "sentences": req.sentences,
            "src_lang": req.src_lang,
            "tgt_lang": req.tgt_lang
        }
        print(f"[DEBUG] Payload sent to translation service: {translation_payload}")
        response = requests.post(TRANSLATION_API_URL, json=translation_payload)
        print(f"[DEBUG] Translation service responded with status: {response.status_code}")
        response.raise_for_status()

        json_response = response.json()
        print(f"[DEBUG] Translation service JSON: {json_response}")
        translations = json_response.get("translations", [])
        print(f"[DEBUG] Extracted translations: {translations}")

    except Exception as e:
        print(f"[ERROR] Failed to call translation service: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}")

    result = []
    for original, translated in zip(req.sentences, translations):
        print(f"[INFO] Processing sentence: '{original}'")
        print(f"[INFO] Transliterating: '{translated}' from Devanagari to {req.target_script}")

        try:
            if req.target_script not in target_script_map:
                raise HTTPException(status_code=400, detail=f"Unsupported target script: {req.target_script}")
            translit = transliterate(translated, DEVANAGARI, target_script_map[req.target_script])
        except Exception as e:
            print(f"[ERROR] Transliteration failed: {e}")
            raise HTTPException(status_code=500, detail=f"Transliteration failed: {e}")

        end_time = time.perf_counter()
        latency_ms = round((end_time - start_time) * 1000, 3)
        translation_cost = round(len(translated) * 0.00005, 6)
        mem_after = process.memory_info().rss
        space_complexity_bytes = mem_after - mem_before
        time_complexity_ops = len(original) * 5

        result.append(TransliterationResponse(
            original=original,
            translated=translated,
            transliterated=translit,
            latency_ms=latency_ms,
            translation_cost=translation_cost,
            space_complexity_bytes=space_complexity_bytes,
            time_complexity_ops=time_complexity_ops
        ))

    print("[INFO] Translation and transliteration complete.")
    return result
