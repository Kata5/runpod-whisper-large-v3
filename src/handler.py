import base64

import runpod
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, cache_dir="/runpod-volume/"
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id, cache_dir="/runpod-volume/")

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    chunk_length_s=30,
    batch_size=16,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

def handler(job):
    job_input = job["input"]

    audio = base64.b64decode(job_input["audio"])

    result = pipe(audio)
    return result["text"]


runpod.serverless.start({"handler": handler})