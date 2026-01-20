import base64
import hashlib
import io
import json
import os
import tempfile
import threading
from collections import OrderedDict
from typing import Any, Literal

import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from zonos.conditioning import make_cond_dict, supported_language_codes
from zonos.model import Zonos, DEFAULT_BACKBONE_CLS as ZonosBackbone
from zonos.utils import DEFAULT_DEVICE as device


app = FastAPI(title="Zonos API", version="0.1.0")

CURRENT_MODEL_TYPE = None
CURRENT_MODEL = None

SPEAKER_CACHE: OrderedDict[str, torch.Tensor] = OrderedDict()
SPEAKER_CACHE_MAX = 8

SILENCE_PREFIX_CODES = None
SILENCE_PREFIX_PATH = "assets/silence_100ms.wav"

MODEL_LOCK = threading.Lock()


def _load_audio_bytes(data: bytes) -> tuple[torch.Tensor, int]:
    try:
        return torchaudio.load(io.BytesIO(data))
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            f.write(data)
            f.flush()
            return torchaudio.load(f.name)


def _hash_audio_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_float_list(value: str | None, name: str) -> list[float] | None:
    if value is None or value == "":
        return None
    try:
        data = json.loads(value)
        if isinstance(data, list):
            return [float(x) for x in data]
    except json.JSONDecodeError:
        pass
    try:
        return [float(x) for x in value.split(",") if x.strip() != ""]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {name} list") from exc


def _parse_str_list(value: str | None) -> list[str] | None:
    if value is None or value == "":
        return None
    if value.startswith("["):
        try:
            data = json.loads(value)
            if isinstance(data, list):
                return [str(x) for x in data]
        except json.JSONDecodeError:
            pass
    return [x.strip() for x in value.split(",") if x.strip() != ""]


def _estimate_max_tokens(text: str, speaking_rate: float, buffer_factor: float = 1.5) -> int:
    estimated_phonemes = len(text) * 1.2
    estimated_duration_seconds = estimated_phonemes / max(speaking_rate, 5.0)
    estimated_tokens = int(estimated_duration_seconds * 86 * buffer_factor)
    return max(172, min(estimated_tokens, 86 * 30))


def _load_model_if_needed(model_choice: str) -> Zonos:
    global CURRENT_MODEL, CURRENT_MODEL_TYPE
    if CURRENT_MODEL_TYPE != model_choice:
        if CURRENT_MODEL is not None:
            del CURRENT_MODEL
            torch.cuda.empty_cache()
        CURRENT_MODEL = Zonos.from_pretrained(model_choice, device=device)
        CURRENT_MODEL.requires_grad_(False).eval()
        CURRENT_MODEL_TYPE = model_choice
    return CURRENT_MODEL


def _get_silence_prefix_codes(model: Zonos) -> torch.Tensor:
    global SILENCE_PREFIX_CODES
    if SILENCE_PREFIX_CODES is None:
        wav_prefix, sr_prefix = torchaudio.load(SILENCE_PREFIX_PATH)
        wav_prefix = wav_prefix.mean(0, keepdim=True)
        wav_prefix = model.autoencoder.preprocess(wav_prefix, sr_prefix)
        wav_prefix = wav_prefix.to(device, dtype=torch.float32)
        SILENCE_PREFIX_CODES = model.autoencoder.encode(wav_prefix.unsqueeze(0))
    return SILENCE_PREFIX_CODES


def _cache_speaker_embedding(cache_key: str, embedding: torch.Tensor) -> torch.Tensor:
    SPEAKER_CACHE[cache_key] = embedding
    SPEAKER_CACHE.move_to_end(cache_key)
    if len(SPEAKER_CACHE) > SPEAKER_CACHE_MAX:
        SPEAKER_CACHE.popitem(last=False)
    return embedding


def _get_speaker_embedding(model: Zonos, wav: torch.Tensor, sr: int, cache_key: str) -> torch.Tensor:
    if cache_key in SPEAKER_CACHE:
        SPEAKER_CACHE.move_to_end(cache_key)
        return SPEAKER_CACHE[cache_key]
    embedding = model.make_speaker_embedding(wav, sr).to(device, dtype=torch.float16)
    return _cache_speaker_embedding(cache_key, embedding)


def _to_wav_bytes(wav: torch.Tensor, sample_rate: int) -> bytes:
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    buffer = io.BytesIO()
    torchaudio.save(buffer, wav, sample_rate, format="wav")
    buffer.seek(0)
    return buffer.read()


def _prepare_vqscore_8(vqscore_8: list[float] | None) -> list[float] | None:
    if vqscore_8 is None:
        return None
    if len(vqscore_8) == 1:
        return vqscore_8 * 8
    return vqscore_8


def _run_generation(
    *,
    model_choice: str,
    text: str,
    language: str,
    speaker_audio_bytes: bytes | None,
    prefix_audio_bytes: bytes | None,
    use_silence_prefix: bool,
    cfg_scale: float,
    top_p: float,
    top_k: int,
    min_p: float,
    linear: float,
    confidence: float,
    quadratic: float,
    seed: int,
    randomize_seed: bool,
    max_new_tokens: int | None,
    speaking_rate: float,
    emotion: list[float] | None,
    vqscore_8: list[float] | None,
    fmax: float,
    pitch_std: float,
    dnsmos_ovrl: float,
    speaker_noised: bool,
    unconditional_keys: list[str] | None,
    disable_torch_compile: bool,
) -> tuple[int, torch.Tensor, int]:
    with MODEL_LOCK, torch.inference_mode():
        model = _load_model_if_needed(model_choice)

        if randomize_seed:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        torch.manual_seed(int(seed))

        speaker_embedding = None
        if speaker_audio_bytes is not None and "speaker" not in (unconditional_keys or []):
            cache_key = _hash_audio_bytes(speaker_audio_bytes)
            wav, sr = _load_audio_bytes(speaker_audio_bytes)
            wav = wav.mean(0, keepdim=True)
            speaker_embedding = _get_speaker_embedding(model, wav, sr, cache_key)

        audio_prefix_codes = None
        if prefix_audio_bytes is not None:
            wav_prefix, sr_prefix = _load_audio_bytes(prefix_audio_bytes)
            wav_prefix = wav_prefix.mean(0, keepdim=True)
            wav_prefix = model.autoencoder.preprocess(wav_prefix, sr_prefix)
            wav_prefix = wav_prefix.to(device, dtype=torch.float32)
            audio_prefix_codes = model.autoencoder.encode(wav_prefix.unsqueeze(0))
        elif use_silence_prefix:
            audio_prefix_codes = _get_silence_prefix_codes(model)

        if max_new_tokens is None:
            max_new_tokens = _estimate_max_tokens(text, speaking_rate)

        vqscore_8 = _prepare_vqscore_8(vqscore_8)
        cond_dict = make_cond_dict(
            text=text,
            language=language,
            speaker=speaker_embedding,
            emotion=emotion or None,
            vqscore_8=vqscore_8 or None,
            fmax=fmax,
            pitch_std=pitch_std,
            speaking_rate=speaking_rate,
            dnsmos_ovrl=dnsmos_ovrl,
            speaker_noised=speaker_noised,
            device=device,
            unconditional_keys=set(unconditional_keys or ["emotion"]),
        )
        conditioning = model.prepare_conditioning(cond_dict)

        codes = model.generate(
            prefix_conditioning=conditioning,
            audio_prefix_codes=audio_prefix_codes,
            max_new_tokens=max_new_tokens,
            cfg_scale=cfg_scale,
            batch_size=1,
            sampling_params=dict(
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                linear=linear,
                conf=confidence,
                quad=quadratic,
            ),
            progress_bar=False,
            disable_torch_compile=disable_torch_compile,
        )

        wav_out = model.autoencoder.decode(codes).cpu().detach()
        if wav_out.dim() == 2 and wav_out.size(0) > 1:
            wav_out = wav_out[0:1, :]
        sr_out = model.autoencoder.sampling_rate
        return sr_out, wav_out.squeeze(0), int(seed)


class GenerateRequest(BaseModel):
    model_choice: str = "Zyphra/Zonos-v0.1-transformer"
    text: str
    language: str = "en-us"
    speaker_audio_base64: str | None = None
    prefix_audio_base64: str | None = None
    use_silence_prefix: bool = True
    cfg_scale: float = 2.0
    top_p: float = 0.0
    top_k: int = 0
    min_p: float = 0.0
    linear: float = 0.0
    confidence: float = 0.0
    quadratic: float = 0.0
    seed: int = 0
    randomize_seed: bool = True
    max_new_tokens: int | None = None
    speaking_rate: float = 15.0
    emotion: list[float] | None = None
    vqscore_8: list[float] | None = None
    fmax: float = 24000.0
    pitch_std: float = 45.0
    dnsmos_ovrl: float = 4.0
    speaker_noised: bool = False
    unconditional_keys: list[str] | None = None
    disable_torch_compile: bool = False
    response_format: Literal["json", "wav"] = "json"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "device": str(device)}


@app.get("/models")
def list_models() -> dict[str, Any]:
    supported_models = []
    if "transformer" in ZonosBackbone.supported_architectures:
        supported_models.append("Zyphra/Zonos-v0.1-transformer")
    if "hybrid" in ZonosBackbone.supported_architectures:
        supported_models.append("Zyphra/Zonos-v0.1-hybrid")
    return {"models": supported_models}


@app.get("/languages")
def list_languages() -> dict[str, Any]:
    return {"languages": supported_language_codes}


@app.post("/generate")
def generate(
    text: str = Form(...),
    language: str = Form("en-us"),
    model_choice: str = Form("Zyphra/Zonos-v0.1-transformer"),
    speaker_audio: UploadFile | None = File(None),
    prefix_audio: UploadFile | None = File(None),
    use_silence_prefix: bool = Form(True),
    cfg_scale: float = Form(2.0),
    top_p: float = Form(0.0),
    top_k: int = Form(0),
    min_p: float = Form(0.0),
    linear: float = Form(0.0),
    confidence: float = Form(0.0),
    quadratic: float = Form(0.0),
    seed: int = Form(0),
    randomize_seed: bool = Form(True),
    max_new_tokens: int | None = Form(None),
    speaking_rate: float = Form(15.0),
    emotion: str | None = Form(None),
    vqscore_8: str | None = Form(None),
    fmax: float = Form(24000.0),
    pitch_std: float = Form(45.0),
    dnsmos_ovrl: float = Form(4.0),
    speaker_noised: bool = Form(False),
    unconditional_keys: str | None = Form(None),
    disable_torch_compile: bool = Form(False),
    response_format: Literal["json", "wav"] = Form("json"),
):
    if language not in supported_language_codes:
        raise HTTPException(status_code=400, detail="Unsupported language code")

    speaker_bytes = speaker_audio.file.read() if speaker_audio is not None else None
    prefix_bytes = prefix_audio.file.read() if prefix_audio is not None else None

    emotion_list = _parse_float_list(emotion, "emotion")
    vqscore_list = _parse_float_list(vqscore_8, "vqscore_8")
    uncond_list = _parse_str_list(unconditional_keys)

    sr_out, wav_out, seed_out = _run_generation(
        model_choice=model_choice,
        text=text,
        language=language,
        speaker_audio_bytes=speaker_bytes,
        prefix_audio_bytes=prefix_bytes,
        use_silence_prefix=use_silence_prefix,
        cfg_scale=cfg_scale,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        linear=linear,
        confidence=confidence,
        quadratic=quadratic,
        seed=seed,
        randomize_seed=randomize_seed,
        max_new_tokens=max_new_tokens,
        speaking_rate=speaking_rate,
        emotion=emotion_list,
        vqscore_8=vqscore_list,
        fmax=fmax,
        pitch_std=pitch_std,
        dnsmos_ovrl=dnsmos_ovrl,
        speaker_noised=speaker_noised,
        unconditional_keys=uncond_list,
        disable_torch_compile=disable_torch_compile,
    )

    wav_bytes = _to_wav_bytes(wav_out, sr_out)
    if response_format == "wav":
        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={"X-Seed": str(seed_out)},
        )
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    return JSONResponse({"sample_rate": sr_out, "audio_base64": audio_b64, "seed": seed_out})


@app.post("/generate-json")
def generate_json(payload: GenerateRequest):
    if payload.language not in supported_language_codes:
        raise HTTPException(status_code=400, detail="Unsupported language code")

    speaker_bytes = base64.b64decode(payload.speaker_audio_base64) if payload.speaker_audio_base64 else None
    prefix_bytes = base64.b64decode(payload.prefix_audio_base64) if payload.prefix_audio_base64 else None

    sr_out, wav_out, seed_out = _run_generation(
        model_choice=payload.model_choice,
        text=payload.text,
        language=payload.language,
        speaker_audio_bytes=speaker_bytes,
        prefix_audio_bytes=prefix_bytes,
        use_silence_prefix=payload.use_silence_prefix,
        cfg_scale=payload.cfg_scale,
        top_p=payload.top_p,
        top_k=payload.top_k,
        min_p=payload.min_p,
        linear=payload.linear,
        confidence=payload.confidence,
        quadratic=payload.quadratic,
        seed=payload.seed,
        randomize_seed=payload.randomize_seed,
        max_new_tokens=payload.max_new_tokens,
        speaking_rate=payload.speaking_rate,
        emotion=payload.emotion,
        vqscore_8=payload.vqscore_8,
        fmax=payload.fmax,
        pitch_std=payload.pitch_std,
        dnsmos_ovrl=payload.dnsmos_ovrl,
        speaker_noised=payload.speaker_noised,
        unconditional_keys=payload.unconditional_keys,
        disable_torch_compile=payload.disable_torch_compile,
    )

    wav_bytes = _to_wav_bytes(wav_out, sr_out)
    if payload.response_format == "wav":
        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={"X-Seed": str(seed_out)},
        )
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    return JSONResponse({"sample_rate": sr_out, "audio_base64": audio_b64, "seed": seed_out})


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port)
