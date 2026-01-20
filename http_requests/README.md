# Zonos API HTTP requests

This folder contains a VS Code REST Client file to call the FastAPI server in `api_server.py`.

## Quick start

1) Start the API server:

```
python api_server.py
```

2) Run a request (saves WAV):

Open `http_requests/zonos.http` in VS Code and run the request you want. The WAV response is saved to `outputs/out.wav`.

## VS Code REST Client (.http)

Open `http_requests/zonos.http` in VS Code and run the request you want. The WAV response is saved to `outputs/out.wav`.

## JSON request notes

The JSON endpoint accepts base64 audio. You can generate a base64 string like this:

```
python - <<'PY'
import base64
with open('/path/to/speaker.wav', 'rb') as f:
    print(base64.b64encode(f.read()).decode('ascii'))
PY
```

Set the result into `speaker_audio_base64` or `prefix_audio_base64` in the JSON request body in `zonos.http`.
