# Speedup Notes

This document tracks performance-related changes and additions made in this repo.

## Precision and dtype changes
- `zonos/model.py`: Model weights are loaded on the target device in `torch.float16` (`Zonos.from_local`).  
- `zonos/model.py`: `make_speaker_embedding` returns the LDA speaker embedding in `float16` (was `bfloat16`).  
- `zonos/backbone/_torch.py`: KV cache allocation uses `torch.float16` (was `bfloat16`).  
- `zonos/backbone/_mamba_ssm.py`: KV cache allocation uses `torch.float16` (was `bfloat16`).  
- `zonos/autoencoder.py`: DAC decode already runs under autocast with `float16` on GPU.

## Speaker embedding fast path
- `api_server.py`: New `/speaker-embedding` endpoint returns a speaker embedding as a flat list.  
- `api_server.py`: `/generate` and `/generate-json` accept `speaker_embedding` and skip the speaker encoder when provided.  
- `http_requests/zonos.http`: Added requests to fetch and use speaker embeddings.

## Caching
- `api_server.py`: LRU speaker embedding cache keyed by SHA-256 of audio bytes (`SPEAKER_CACHE`, size 8).  
- `api_server.py`: Silence prefix codes cached in memory (`SILENCE_PREFIX_CODES`).  
- `speaker_cloning.py`: Resamplers cached per sample rate via `@cache` on `_get_resampler`.

## Logging for validation
- `api_server.py`: Per-request log lines include speaker source (`embedding` vs `audio`), input details, and timing breakdown (speaker/prefix/conditioning/generation).  

## Notes
- Using `speaker_embedding` saves only the speaker encoder time (~tens of ms).  
  The autoregressive decode loop (`model.generate`) remains the dominant cost.
