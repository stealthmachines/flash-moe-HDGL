# WARNING: This repo uses Abitrary Precision, which has been known to break computers.  My suspicion is that this may be caused by echo or feedback.  This shouldn't scare you away if you're a researcher, rather it shows the computational power almost transcends its own virtual environment by breaking hardware.  Cool.  But yeah, be careful if you use this code, as damage has been known to happen.  To solve this, I have developed this over here which may help in the future to isolate the echo chambers to a form which is contained or controlled - https://github.com/stealthmachines/AnalogContainer1  Similarly, any given virtual environment may help to end smoked hardware, but further research is required to this end at this time.

# Flash-MoE + HDGL-28: Hybrid Inference on a 397B Parameter Model

> **[Read the original paper](paper/flash_moe.pdf)** — Full technical details, 90+ experiments, and the story of how an AI and a human built the Flash-MoE engine in 24 hours.

Pure C/Metal inference engine running **Qwen3.5-397B-A17B** on a MacBook Pro with 48GB RAM at **4.4+ tokens/second**, now extended with **HDGL-28** — a hybrid Mixture-of-Experts routing system that bridges deterministic gate-score routing with a temporal, sequence-aware lattice signal.

The entire 209GB model streams from SSD through a custom Metal compute pipeline. No Python. No frameworks. Just C, Objective-C, hand-tuned Metal shaders, and now an arbitrary-precision lattice.

---

## What's New: HDGL-28 Hybrid Routing

### The Problem with Pure MoE Gating

Standard MoE routing is stateless. At each layer, the gate network scores all 512 experts against the current hidden vector and picks the top K. It has no memory of which experts have been active across the sequence, no awareness of temporal patterns, and no mechanism to reinforce expert pathways that have been productive.

### The HDGL-28 Solution: Left Brain / Right Brain

HDGL-28 adds a second, parallel routing signal derived from a **BootloaderZ V6.0 Arbitrary Precision Architecture (APA) lattice** — a dynamically evolving data structure that tracks sequence-level expert activation patterns across all 60 layers independently.

The analogy is deliberate:

- **Left brain (MoE gate)** — analytical, deterministic, weight-driven. Scores experts from the current token's hidden state. Fast and precise but myopic.
- **Right brain (HDGL lattice)** — temporal, holistic, history-driven. Tracks which expert pathways have been reinforcing across the sequence. Slow to form opinions but sensitive to patterns the gate can't see.

At every layer's gating decision, both signals are blended before top-K selection:

```
gate_scores[hdgl_expert] += alpha        # boost HDGL-preferred expert
cpu_softmax(gate_scores, NUM_EXPERTS)    # renormalise
cpu_topk(gate_scores, NUM_EXPERTS, K)   # then select
```

The default `alpha = 0.20` means the lattice contributes a 20% nudge. The MoE gate still drives the primary decision; the lattice breaks ties and reinforces productive pathways it has observed across the sequence.

### How the Lattice Works

The HDGL-28 lattice is a chunked array of **Slot4096** structures — each an arbitrary-precision floating-point accumulator with a 4096-bit mantissa, φ-scaled dynamic base, and MPI-backed exponent. At startup (`--hdgl`):

1. **4096 instances × 4 slots** are allocated in 1M-slot chunks (lazy, on demand)
2. **50 seeding steps** of `lattice_step_cpu` run the prismatic recursion: each slot accumulates `sqrt(φ^i · fib[i] · prime[i] · 2^i · ω) · val^(i%7+1)` scaled by a time-evolving oscillator
3. **`g_hdgl_history[60]`** — one `HDGL_History` per transformer layer — tracks `last_feedback` and `last_expert_id` across tokens

At each gating step, a token key is encoded with a phi-tau path function and projected through the Spiral8 primary+mirror strands into lattice coordinates. Slot state (`charge/entropy/tension`) is combined with per-layer history to produce a temporal feedback signal that influences expert choice. The history is updated after each step, so the next token's routing is sequence-conditioned.

### New Files

```
metal_infer/
  hdgl_bootloaderz.h     # Unified header: APA types, lattice API, router types,
                         #   mpi_update_from_legacy inline, HDGL_ShaderDims
  hdgl_bootloaderz.c     # BootloaderZ V6.0 APA library (main() removed):
                         #   MPI, Slot4096, ap_add/copy/normalize/shift,
                         #   lattice_init/step/fold/free, hdgl_get_slot_state,
                         #   free_apa_constants (idempotent, flag-guarded)
  hdgl_router.h          # Public router API
  hdgl_router.c          # Recursive temporal token router:
                         #   hdgl_router_init, route_token_recursive,
                         #   route_tokens_recursive, hdgl_get_packed_dims
```

### Memory Safety

All four bugs present in the original `bootloaderZ.c` port were found and fixed under AddressSanitizer before merge:

| Bug | Location | Fix |
|-----|----------|-----|
| Aliased MPI UAF in `ap_copy` | `hdgl_bootloaderz.c` | Zero MPI word pointers after shallow copy |
| Uninitialized slots freed in `lattice_free` | `hdgl_bootloaderz.c` | `calloc` instead of `malloc` for chunk slots |
| Uninitialized stack `B_aligned` freed | `hdgl_bootloaderz.c` | `= {0}` zero-initialization |
| `APA_CONST_PHI/PI` leaked at process exit | `hdgl_bootloaderz.c` | `free_apa_constants()` with init-flag guard, called from `lattice_free` |

**55/55 tests pass with zero memory errors, zero UAF, zero leaks** under full ASAN + LeakSanitizer.

---

## Results

![Progress](progress.png)

| Configuration | tok/s | Quality | Notes |
|--------------|-------|---------|-------|
| 4-bit experts, FMA kernel | **4.36** | Excellent | Baseline. Full tool calling. 209GB on disk. |
| 4-bit experts, baseline | 3.90 | Excellent | Before FMA kernel optimization. |
| 2-bit experts, trust OS | 5.74 | Good* | 120GB on disk. *Breaks JSON/tool calling. |
| 2-bit peak single token | 7.05 | Good* | Warm cache burst. *Not suitable for tool use. |
| **4-bit + HDGL-28** | **~4.3** | Excellent | Hybrid routing. Small per-token CPU overhead. |

*2-bit quantization produces `\name\` instead of `"name"` in JSON output, making tool calling unreliable. 4-bit is the production configuration.*

*HDGL-28 overhead is the cost of `route_token_recursive` per layer per token — approximately 60 × phi-tau encode + spiral projection + history update — negligible against the 2.4ms SSD I/O per layer.*

---

## Hardware

- **Machine**: MacBook Pro, Apple M3 Max
- **Chip**: 16-core CPU (12P + 4E), 40-core GPU, 16-core ANE
- **Memory**: 48 GB unified (~400 GB/s bandwidth)
- **SSD**: 1TB Apple Fabric, **17.5 GB/s sequential read** (measured)
- **macOS**: 26.2 (Darwin 25.2.0)

---

## Quick Start

```bash
cd metal_infer
make infer

# Standard 4-bit inference
./infer --prompt "Explain quantum computing" --tokens 200

# Hybrid HDGL-28 inference (default alpha=0.20)
./infer --hdgl --prompt "Explain quantum computing" --tokens 200

# HDGL with lighter lattice influence
./infer --hdgl --hdgl-alpha 0.10 --prompt "Explain quantum computing" --tokens 200

# HDGL with stronger temporal signal
./infer --hdgl --hdgl-alpha 0.35 --prompt "Explain quantum computing" --tokens 200

# Pre-seed lattice once, then load it for fast startup
make hdgl-preseed
./infer --hdgl --hdgl-load hdgl_lattice.bin --prompt "Explain quantum computing" --tokens 200

# Interactive chat (serve mode required)
make chat
./infer --serve 8000 --hdgl &
./chat --hdgl

# 2-bit mode (faster, breaks tool calling)
./infer --2bit --prompt "Explain quantum computing" --tokens 200
```

---

## Build

```bash
cd metal_infer
make          # builds metal_infer + infer + chat
make infer    # inference engine only
make chat     # chat TUI only
make clean
```

The build requires macOS 14+ with Xcode command line tools. Metal shaders are compiled at runtime from source via `MTLDevice newLibraryWithSource:` — no offline shader compiler needed.

---

## Full Command Reference

### `./infer`

**Model paths** (auto-detected relative to executable if not specified):

| Flag | Default | Description |
|------|---------|-------------|
| `--model PATH` | `model_weights.bin` | Model path |
| `--weights PATH` | auto | `model_weights.bin` path |
| `--manifest PATH` | auto | `model_weights.json` tensor manifest |
| `--vocab PATH` | auto | `vocab.bin` vocabulary |
| `--prompt-tokens PATH` | — | Pre-tokenized prompt binary |
| `--prompt TEXT` | — | Prompt text (requires `encode_prompt.py`) |

**Generation:**

| Flag | Default | Description |
|------|---------|-------------|
| `--tokens N` | 20 | Max tokens to generate |
| `--k N` | 4 | Active experts per layer |
| `--think-budget N` | 2048 | Max thinking tokens before forcing `</think>` (0 = unlimited) |

**Quantization:**

| Flag | Description |
|------|-------------|
| `--2bit` | Use 2-bit quantized experts (`packed_experts_2bit/`). Faster, breaks JSON/tool calling. |

**HDGL-28 Hybrid Routing:**

| Flag | Default | Description |
|------|---------|-------------|
| `--hdgl` | off | Enable HDGL-28 BootloaderZ APA lattice backend with sign-magnitude ternary kernel and recursive temporal token routing |
| `--hdgl-alpha N` | 0.20 | Blend weight for the HDGL routing signal. `0.0` = pure MoE gate. `1.0` = pure HDGL routing. Recommended range: `0.10`–`0.35`. |
| `--hdgl-load FILE` | — | Load pre-seeded lattice state from FILE. If load fails, startup falls back to seeded initialization. |

**Compute paths:**

| Flag | Description |
|------|-------------|
| `--gpu-linear` | Fused GPU GatedDeltaNet path (default) |
| `--cpu-linear` | Disable fused GPU delta-net, use CPU/hybrid linear path |
| `--skip-linear` | Skip linear attention entirely |

**Caching:**

| Flag | Default | Description |
|------|---------|-------------|
| `--cache-entries N` | 0 | Expert LRU cache size (0 = disabled, trust OS) |
| `--malloc-cache N` | 0 | Malloc expert cache entries (e.g. 2581 = 17GB for ~80% hit rate) |

**Diagnostics:**

| Flag | Description |
|------|-------------|
| `--timing` | Per-layer timing breakdown (attention, routing, I/O, expert forward) |
| `--freq` | Expert frequency tracking and analysis report at end |
| `--cache-telemetry` | Cold vs eviction misses and reuse distance reporting |
| `--predict` | Enable temporal expert prediction (prefetch during CMD1 wait) |
| `--collect-routing FILE` | Log routing data to binary file for predictor training |

**Server:**

| Flag | Description |
|------|-------------|
| `--serve PORT` | Run OpenAI-compatible HTTP server on PORT |

### `./chat`

| Flag | Default | Description |
|------|---------|-------------|
| `--port N` | 8000 | Inference server port |
| `--max-tokens N` | 8192 | Max response tokens |
| `--show-think` | off | Show `<think>` blocks (displayed dimmed) |
| `--resume ID` | — | Resume a previous session by ID |
| `--sessions` | — | List all saved sessions |
| `--hdgl` | off | Print HDGL-28 banner (lattice mode is controlled on the server side via `./infer --hdgl`) |
| `--help` | — | Show usage |

**In-session commands:**

```
/quit      Exit
/exit      Exit
/clear     Clear conversation history
/sessions  List saved sessions
```

Sessions are saved automatically to `~/.flash-moe/sessions/<id>.jsonl`.

---

## Architecture

The model has 60 transformer layers: 45 GatedDeltaNet (linear attention) + 15 standard full attention. Each layer has 512 experts, of which K=4 are activated per token (plus one shared expert). Hidden dimension is 4096.

### Pipeline Per Layer (4.28ms average at 4-bit)

```
CMD3(prev) → CMD1: attention projections + delta-net  [1.22ms GPU]
           → CPU: flush results                       [0.01ms CPU]
           → CMD2: o_proj + norm + routing + shared    [0.55ms GPU]
           → CPU: softmax + HDGL blend + topK          [~0.003ms + HDGL overhead]
           → I/O: parallel pread K=4 experts           [2.41ms SSD]
           → CMD3: expert forward + combine + norm     [0.04ms encode, DEFERRED]
```

With `--hdgl`, the routing step expands to:

```
           → CPU: softmax(gate_scores)
                route_token_recursive(token_key, &history[layer])
                  gate_scores[hdgl_expert] += alpha
                  softmax(gate_scores)
                  topK(gate_scores, K)
```

### Key Techniques

1. **SSD Expert Streaming** — Expert weights (209GB at 4-bit) are read from NVMe SSD on demand via parallel `pread()` with GCD dispatch groups. Only the K=4 active experts per layer are loaded (~6.75MB each). The OS page cache manages caching — no custom cache needed ("Trust the OS" principle).

2. **FMA-Optimized Dequant Kernel** — The inner loop rearranges `(nibble * scale + bias) * x` to `fma(nibble, scale*x, bias*x)`, letting the GPU fused multiply-add unit do dequant+multiply in one instruction. 12% faster than the naive formulation.

3. **Metal Compute Shaders** — Hand-written Metal kernels for 4-bit/2-bit dequant matvec, fused SwiGLU, RMS normalization, batched GPU attention, GPU RoPE, MoE combine + residual, and the new HDGL-28 sign-magnitude ternary FMA kernel (`sign_magnitude_ternary_fma`).

4. **Sign-Magnitude Ternary FMA** — The HDGL-28 Metal kernel treats weights and activations as sign-magnitude values. The inner product reduces to a sum of `|w|·|x|` products with a per-element ±1 sign, matching the BootloaderZ V6.0 ternary arithmetic model. Active when `--hdgl` is set and the expert dispatch selects `matvec_hdgl`.

5. **Deferred GPU Expert Compute** — CMD3 (expert forward pass) is submitted without waiting. The GPU executes it while the CPU prepares the next layer.

6. **Accelerate BLAS for Linear Attention** — The GatedDeltaNet recurrence uses `cblas_sscal`, `cblas_sgemv`, and `cblas_sger` for the 64-head × 128×128 state matrix update. 64% faster than scalar code.

7. **Trust the OS** — No custom expert cache. The OS page cache (~35GB) manages expert data caching via standard LRU, achieving ~71% hit rate naturally.

### HDGL-28 Architecture Detail

```
BootloaderZ V6.0 Lattice
  4096 instances × 4 slots = 16,384 Slot4096 accumulators
  Chunked in 1M-slot HDGLChunk blocks (lazy allocation)
  Each Slot4096: 4096-bit mantissa (64 × uint64_t words)
                 φ-scaled dynamic base
                 MPI-backed exponent (arbitrarily wide)
                 GOI/GUZ state flags

Seeding (50 steps of lattice_step_cpu):
  For each slot i:
    val = ap_to_double(slot[i])
    r   = sqrt(φ^(i%16) · fib[i%16] · prime[i%16] · 2^(i%16) · ω) · val^((i%7)+1)
    slot[i] += r · tick

Per-token routing:
  Token key  -> phi_tau(text) depth-encoded scalar
  Strand     -> Spiral8 primary + mirror counter-rotation
  Feedback   -> slot state + temporal history blend
  expert_id  -> projected lattice coord + blended signal mod NUM_EXPERTS
  H updated  -> next token routing is history-conditioned

Per-layer history:
  g_hdgl_history[60] — one HDGL_History per transformer layer
  Early layers (0–14, full attention) develop independent routing biases
  from later layers (15–59, linear attention) — structural vs semantic
```

---

## Project Structure

```
metal_infer/
  infer.m                  # Complete inference engine (~7150 lines)
  shaders.metal            # Metal compute kernels (~1340 lines, incl. HDGL kernel)
  chat.m                   # Interactive chat TUI with tool calling
  tokenizer.h              # C BPE tokenizer (single-header)
  main.m                   # MoE-only benchmark
  Makefile                 # Build system
  hdgl_bootloaderz.h       # HDGL-28 unified header
  hdgl_bootloaderz.c       # BootloaderZ V6.0 APA lattice library
  hdgl_router.h            # Recursive temporal router API
  hdgl_router.c            # Recursive temporal router implementation
  hdgl_lattice_generator.c # Offline lattice pre-seeder (writes hdgl_lattice.bin)
  bootloaderZ.c            # Original BootloaderZ V6.0 source (reference)
  extract_weights.py       # Creates model_weights.bin from safetensors
  repack_experts_2bit.py   # 4-bit → 2-bit expert requantization
  train_predictor.py       # Expert routing prediction analysis
  model_weights.bin        # Non-expert weights (5.5GB, mmap'd)
  model_weights.json       # Tensor manifest
  vocab.bin                # Vocabulary for token decoding
  tokenizer.bin            # Pre-exported BPE tokenizer data

repack_experts.py          # 4-bit expert packing from safetensors
progress.py                # Results visualization
results.tsv                # Experiment log
```

---

## What We Tried (and What Worked)

### Kept
| Approach | Result | Impact |
|----------|--------|--------|
| FMA dequant kernel | GPU compute -12% | **+12% tok/s** |
| Trust OS page cache | Deleted Metal LRU → +38% | **Foundational** |
| GPU combine+norm in CMD3 | Eliminates CPU round-trip | **Pipeline** |
| BLAS delta-net (Accelerate) | cpu_attn 0.78→0.28ms | **+64% attn** |
| F_NOCACHE for 2-bit | +3% from avoiding page thrash | **2-bit only** |
| GPU fused attention (RoPE) | +2% for full-attn layers | **Small** |
| C BPE tokenizer | 180ms vs 3500ms startup | **20x startup** |
| Deferred CMD3 execution | GPU/CPU overlap | **Pipeline** |
| **HDGL-28 hybrid routing** | **Temporal expert memory** | **New** |

### Discarded (58+ experiments, highlights)
| Approach | Result | Why |
|----------|--------|-----|
| LZ4 expert compression | -13% | Decompress overhead > warm cache savings |
| F_RDADVISE prefetch | net 0% | Unified memory: SSD DMA slows GPU -73% |
| Temporal expert prediction | -18% | 25% hit rate, SSD bandwidth waste |
| MLP routing predictor | 31% accuracy | Worse than temporal baseline |
| GPU LUT dequant kernel | -2% | Indirect register access serializes |
| GPU private buffer compression | -20% pipeline | Blit cost 4×7MB > matvec savings |
| Spin-poll GPU wait | -23% | CPU thermal competes with GPU |
| Expert file clustering | 0% | NVMe ignores scatter at 7MB granularity |
| dispatch_io | -70% | dispatch_data management overhead |
| mmap expert files | -5x | Per-page fault overhead on cold data |
| Speculative early routing | -38% | Cache pollution + overhead |
| MTP speculative decoding | break-even | MoE I/O scales per-token |

---

## Safety

This is a primary development machine. The engine explicitly controls memory:
- Non-expert weights: 5.5GB (mmap'd, read-only)
- Metal scratch buffers: ~200MB
- HDGL-28 lattice: ~2MB (4096 × 4 slots, lazily allocated)
- Total: ~6GB, leaving 42GB for OS + page cache
- No OOM risk. Expert data streams from SSD on demand.
- No custom caches. Trust the OS.

The HDGL-28 lattice is allocated once at startup and freed via `lattice_free` which also calls `free_apa_constants()` to release the PHI/PI precision constants. All memory paths verified clean under AddressSanitizer.

---

## Credits

Original Flash-MoE engine by [danveloper](https://github.com/danveloper/flash-moe).

HDGL-28 hybrid routing extension by [stealthmachines](https://github.com/stealthmachines/flash-moe-HDGL), built on the [BootloaderZ V6.0](https://zchg.org) arbitrary precision architecture.

# LICENSING - ALL LICENSING, AS APPLICABLE, for 'flash-moe' original files retain their original licensing.

All files diverging at time of publication of this repo remain property of ZCHG.org persuant but not limited to - https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440

This repo does not have the authority to usurp its parent licensing, that of https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440

# https://zchg.org
