# Differential Correction in Wavelet Domain (DCW)

ACE-Step-1.5 supports an **opt-in sampler-side correction** from:

> **Elucidating the SNR-t Bias of Diffusion Probabilistic Models**
> Meng Yu, Lei Sun, Jianhao Zeng, Xiangxiang Chu, Kun Zhan. **CVPR 2026**.
> arXiv:[2604.16044](https://arxiv.org/abs/2604.16044) · Reference code: [AMAP-ML/DCW](https://github.com/AMAP-ML/DCW)

DCW is training-free, adds negligible compute, and has been validated across a
wide range of diffusion models — including **FLUX**, which shares ACE-Step's
flow-matching formulation.

## What it does

During **training**, every noised sample `x_t` has an SNR that is strictly
coupled to its timestep `t`. At **inference** the sampler steps through a
*predicted* trajectory, so the actual SNR of `x_t` drifts away from what the
network was trained to see at that `t`. This *SNR-t bias* accumulates along
the denoising chain and hurts quality — most visibly at intermediate / high
timesteps.

The authors' key observation is that diffusion models reconstruct
**low-frequency content before high-frequency content** during the reverse
process. Therefore the SNR-t drift affects different frequency bands
differently, and the correction should be applied **per-band** (wavelet
domain) rather than directly on the latent.

### Per-step update

Given the current latent `x`, velocity prediction `v`, and timestep `t_curr`:

```text
x_next   = sampler_step(x, v, t_curr)      # normal Euler / Heun / flow-matching
denoised = x - v * t_curr                  # predicted clean sample x_0

xL, xH   = DWT(x_next)
yL, yH   = DWT(denoised)

xL       = xL + s * (xL - yL)              # "low" mode   (Eq. 18 / 20)
xH[j]    = xH[j] + s * (xH[j] - yH[j])     # "high" mode
x_next   = IDWT(xL, xH)
```

For flow-matching ACE-Step, the scaler is modulated by the current noise
level: `s = t_curr * dcw_scaler`. So a constant user-facing `dcw_scaler`
starts at its nominal value near `t=1` and decays to zero as `t → 0`.

ACE-Step latents are 1-D temporal tensors of shape `[B, T, 64]` at 25 Hz,
so DCW uses a 1-D DWT along the `T` axis.

## Installation

DCW's PyTorch-path wavelet transforms use `pytorch_wavelets` + `PyWavelets`.
Both are declared in `pyproject.toml`, so a normal `uv sync` already
installs them — nothing extra to do. On the MLX path, `haar` runs natively
in MLX (no Python wavelet deps needed); non-Haar bases bridge to the same
`pytorch_wavelets` modules.

If, for some reason, `pytorch_wavelets` is missing from the environment
when DCW is enabled, the pipeline logs a clear warning and falls back to
the normal sampler — no crash.

## Parameters

All DCW parameters live on `GenerationParams` in `acestep/inference.py` and
are forwarded through the generation handler chain into the base model's
`generate_audio`.

| Field | Type | Default | Description |
|---|---|---|---|
| `dcw_enabled` | `bool` | `True` | Master switch. Set to `False` for a clean A/B against the uncorrected sampler. |
| `dcw_mode` | `str` | `"low"` | One of `"low"`, `"high"`, `"double"`, `"pix"`. |
| `dcw_scaler` | `float` | `0.02` | Low-band correction strength (or the single scaler for `"high"` / `"pix"`). Usable range `0–0.1`. |
| `dcw_high_scaler` | `float` | `0.0` | High-band correction strength (used only when `dcw_mode == "double"`). Usable range `0–0.1`. |
| `dcw_wavelet` | `str` | `"haar"` | PyWavelets basis name — e.g. `"haar"`, `"db4"`, `"sym8"`. |

### Mode reference

- **`low`** — Push only the low-frequency band away from the denoised
  estimate. Matches the `dcw_low` code example in the paper's README
  and is our recommended starting point on ACE-Step. Note that the
  reference repo does *not* claim a single "best" mode — the FLUX
  scheduler in AMAP-ML/DCW actually ships with `pix` as the active
  line and `dcw_low` / `dcw_high` commented out — so treat `low` as a
  sensible default, not a canonical one.
- **`high`** — Same but on the high-frequency detail band.
- **`double`** — Both bands, with independent `dcw_scaler` and
  `dcw_high_scaler`. ACE-Step extension; the reference implementation
  does not expose a separate high-band scaler.
- **`pix`** — No wavelet transform; correction is applied directly in
  latent space. This is the mode the reference FLUX scheduler leaves
  uncommented, so it is useful both as an ablation and as a
  "closest-to-reference" baseline.

## Usage example (Python API)

```python
from acestep.inference import GenerationConfig, GenerationParams, generate_music

# `dit_handler` and `llm_handler` come from the standard ACE-Step setup;
# see docs/en/INFERENCE.md for how they're constructed.

params = GenerationParams(
    caption="mellow lo-fi hiphop with jazzy piano",
    duration=30.0,
    inference_steps=32,
    guidance_scale=7.5,
    # Enable DCW:
    dcw_enabled=True,
    dcw_mode="low",
    dcw_scaler=0.02,
    dcw_wavelet="haar",
)
config = GenerationConfig(batch_size=1)
result = generate_music(
    dit_handler=dit_handler,
    llm_handler=llm_handler,
    params=params,
    config=config,
)
```

### Usage example (Gradio UI)

Open the standard Gradio UI, expand **Advanced DiT** → **🧪 DCW – Differential
Correction in Wavelet domain (experimental)**, and tune the four
sliders/dropdowns inside. **Enable DCW** is on by default with
`mode="low"`, `scaler=0.02`, `wavelet="haar"` — uncheck it to A/B
against the uncorrected sampler.

## Recommended starting values

The defaults are intentionally conservative. For informal listening tests
with `inference_steps ∈ {16, 32}`:

- Start with `dcw_mode="low"`, `dcw_scaler=0.02`, `dcw_wavelet="haar"`.
  Usable scaler range is `0–0.1`; going above that tends to introduce
  audible artefacts.
- If the output sounds *over-smoothed*, try `dcw_mode="double"` with
  small values for both bands (e.g. `dcw_scaler=0.02` and
  `dcw_high_scaler=0.01`) to let the sampler recover high-frequency
  detail as well.
- Try different wavelet bases (`db4`, `sym8`) for smoother low-band
  extraction; `haar` is the default and fastest, with a blocky low-pass
  response.

## Scope

The initial integration enables DCW on the **base** PyTorch sampler path
(`acestep/models/base/modeling_acestep_v15_base.py`). The following are
tracked as follow-up work — see issue #1119:

- Propagate the hook to the **turbo**, **sft**, **xl_base**, **xl_sft**,
  **xl_turbo** model variants.
- Port the correction to the **MLX** path for Apple Silicon.
- Expose the parameters in the Gradio UI.

## Citation

```bibtex
@article{yu2026eluci,
  title   = {Elucidating the SNR-t Bias of Diffusion Probabilistic Models},
  author  = {Meng Yu and Lei Sun and Jianhao Zeng and Xiangxiang Chu and Kun Zhan},
  journal = {arXiv preprint arXiv:2604.16044},
  year    = {2026}
}
```
