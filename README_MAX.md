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

## ⚠️ torchcodec must match torch (audio export)

Audio export (MP3/WAV) goes through `torchcodec`, whose compiled `.so`
is ABI-locked to a specific PyTorch minor version. The pin in this repo
is `torchcodec>=0.9.1,<0.11`, which is the only family compatible with
the pinned `torch 2.10`. If you ever see something like:

```
OSError: libtorchcodec_core8.so: undefined symbol: torch_dtype_float4_e2m1fn_x2
```

(buried inside the FFmpeg-version-8 sub-traceback of a
`Could not load libtorchcodec` error), that means a newer `torchcodec`
got resolved and you have to pin it back:

```bash
uv pip install 'torchcodec>=0.9.1,<0.11'
```

Compatibility table: https://github.com/pytorch/torchcodec#installing-torchcodec
(`0.10` ↔ `torch 2.10`, `0.11` ↔ `torch 2.11`, etc.)

You also need a system FFmpeg in `[4, 8]` providing `libavutil` /
`libavcodec` / `libavformat` shared libs. On Ubuntu:

```bash
sudo apt install -y ffmpeg
```

## API server

From the repo root:

```bash
uv run acestep-api --host 0.0.0.0 --port 8001
```

### Default DiT model

This fork pins the **default primary DiT model to
`acestep-v15-xl-turbo`** (the 4B XL turbo variant from the XL series
released by ACE-Step on 2026-04-02; ≥12 GB VRAM with offload, ≥20 GB
recommended). The model is auto-downloaded from
`ACE-Step/acestep-v15-xl-turbo` on first start and cached under
`checkpoints/acestep-v15-xl-turbo/`.

To use a different model without changing code, set
`ACESTEP_CONFIG_PATH` before launching, e.g.:

```bash
ACESTEP_CONFIG_PATH=acestep-v15-xl-base uv run acestep-api --host 0.0.0.0 --port 8001
ACESTEP_CONFIG_PATH=acestep-v15-turbo   uv run acestep-api --host 0.0.0.0 --port 8001  # pre-XL default
```

Per-request override is also supported via the `model` field on the
`/release_task` payload.

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
