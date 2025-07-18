from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
import time

# --------------------
# Model Setup
# --------------------
print("[INFO] Initializing model setup...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
src_lang_default, tgt_lang_default = "eng_Latn", "hin_Deva"

print(f"[INFO] Using device: {DEVICE}")

print("[INFO] Loading model and tokenizer...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"  # Comment this if FlashAttention is not available
).to(DEVICE)

print("[INFO] Model loaded.")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
ip = IndicProcessor(inference=True)

# --------------------
# FastAPI Setup
# --------------------
app = FastAPI(title="BhashaSetu Translation API", version="1.0")

class TranslationRequest(BaseModel):
    sentences: List[str]
    src_lang: str = src_lang_default
    tgt_lang: str = tgt_lang_default

class TranslationResponse(BaseModel):
    translations: List[str]
    latency: float
    model_params_millions: float
    device: str

@app.get("/")
def root():
    return {"message": "BhashaSetu Translation API is running."}

@app.get("/health")
def health_check():
    return {"status": "ok", "device": DEVICE}

@app.post("/translate", response_model=TranslationResponse)
def translate_text(req: TranslationRequest):
    print("[INFO] Received translation request.")
    if not req.sentences:
        print("[ERROR] No input sentences provided.")
        raise HTTPException(status_code=400, detail="No input sentences provided.")

    start_time = time.perf_counter()

    try:
        print("[INFO] Preprocessing input batch...")
        batch = ip.preprocess_batch(req.sentences, src_lang=req.src_lang, tgt_lang=req.tgt_lang)

        print("[INFO] Tokenizing input batch...")
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        print("[INFO] Generating translation...")
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        print("[INFO] Decoding output tokens...")
        decoded = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        print("[INFO] Postprocessing translations...")
        translations = ip.postprocess_batch(decoded, lang=req.tgt_lang)
        latency = round(time.perf_counter() - start_time, 4)

        print("[INFO] Translation completed.")
        return TranslationResponse(
            translations=translations,
            latency=latency,
            model_params_millions=round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            device=DEVICE
        )

    except Exception as e:
        print(f"[ERROR] Translation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")
