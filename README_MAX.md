# README_MAX

Launch commands for this ACE-Step fork on the GPU server, with external access enabled via `0.0.0.0`.

## ⚠️ Required: install flash-attn first

`flash_attn` is in `requirements.txt` / `pyproject.toml` but isn't always installed correctly and must be
installed manually into the project venv. Without it, nano-vllm falls back
to an SDPA path that is incompatible with CUDA-graph capture; capture
aborts at startup with `cudaErrorStreamCaptureInvalidated` and poisons
torch's CUDA Philox RNG, after which **every generation job fails** with:

```
RuntimeError: Offset increment outside graph capture encountered unexpectedly
```

(at `torch.multinomial` in `acestep/llm_inference.py::_sample_tokens`).

Install a prebuilt wheel that matches the Python / torch / CUDA combo
this repo pins (Python 3.12, torch 2.10, CUDA 12.8):

```bash
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl
```

Verify:

```bash
.venv/bin/python -c "import flash_attn; print(flash_attn.__version__)"
```

For a different Python/torch/CUDA combo, browse
[mjun0812/flash-attention-prebuild-wheels releases](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases)
and pick the matching wheel, or build from source via
`uv pip install flash-attn --no-build-isolation` (slow).

## API server

From the repo root:

```bash
uv run acestep-api --host 0.0.0.0 --port 8001
```

Tmux version:

```bash
tmux new -d -s ace-api 'cd /home/ubuntu/ACE-Step-1.5 && uv run acestep-api --host 0.0.0.0 --port 8001'
```

## Simple UI

From the repo root:

```bash
uv run acestep-simple --server-name 0.0.0.0 --port 7861 --api-base-url http://127.0.0.1:8001
```

Tmux version:

```bash
tmux new -d -s ace-ui 'cd /home/ubuntu/ACE-Step-1.5 && uv run acestep-simple --server-name 0.0.0.0 --port 7861 --api-base-url http://127.0.0.1:8001'
```

## Notes

- Binding to `0.0.0.0` exposes the services on the server network interfaces.
- The UI still points its backend to `http://127.0.0.1:8001`, which is correct when both processes run on the same machine.
- Default external ports for this setup are `8001` for the API and `7861` for the UI.
