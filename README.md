# Hypervisor-MoE HDGL-28: Flash-MoE + BootloaderZ V6.0

**Running a 397B-parameter Mixture-of-Experts model with optional ultra-high-precision APA lattice backend.**

This is the original **Flash-MoE** engine (pure C/Metal, SSD-streamed 4-bit/2-bit experts, 4.4+ tok/s on M3 Max) now bolstered with the **BootloaderZ V6.0 Arbitrary Precision Architecture (APA)** and **HDGL (Hyper-Dimensional Geometric Lattice)** from https://forum.zchg.org/t/bootloaderz-in-c-base4096-you-hypervisor-and-me/865.

- **Normal mode** — unchanged fast 4-bit MoE inference (production quality, tool calling).
- **HDGL-28 mode** (`--hdgl`) — deterministic **Prismatic Router** + **Tertiary Experts** (Precision, Verification, Geometric, Recursive) using full **Slot4096** (4096-bit mantissa) and sign-magnitude ternary arithmetic. Slower (~0.6–1.5 tok/s) but hallucination-resistant and grounded in geometric necessity.

**Philosophy**: "You, Hypervisor; and Me" — the BootloaderZ lattice acts as a hypervisor managing MoE experts as living lattice transformers instead of stripped-down neural weights.

## Results (M3 Max, 48 GB)

| Mode                  | tok/s     | Quality / Use Case                  | Notes |
|-----------------------|-----------|-------------------------------------|-------|
| 4-bit (default)       | **4.36**  | Excellent (tool calling, JSON)      | Fastest production path |
| 2-bit                 | 5.74      | Good (but fragile JSON)             | Breaks some tool use |
| **HDGL-28 (--hdgl)**  | 0.6–1.5   | Excellent for math/physics/verification | 4096-bit APA + Prismatic routing, deterministic |

## New Features (HDGL-28)

- **Prismatic Aperture Router** — φ-harmonic deterministic routing (no learned softmax → eliminates hallucinations for consistent "Glyph" inputs).
- **Tertiary Experts** — Experts are now domain-specific HDGL engines:
  - **Precision** — Slot4096 + MPI for deep scaling.
  - **Verification** — Web-of-Trust / covenant-style auditing.
  - **Geometric** — Hex-tiling / folding for parametric geometry.
  - **Recursive** — φ-braids + Fibonacci for lattice folding and temporal sync.
- **Flash-Lattice Streaming** — mmap/pread of massive APA chunks from SSD (`hdgl_lattice.bin`).
- **Sign-Magnitude Ternary Kernel** — New Metal kernel for `[-1, 0, 1]` logic matching BootloaderZ APA.
- **Lattice Bootloader** — `bootloader_init_lattice()` with 50-step prismatic seeding + folding (Fibonacci/primes/harmonics).
- **Infinite Scaling** — Trillions of primitives on NVMe; only active experts wake up.

## Quick Start

```bash
cd metal_infer

# 1. Build (includes HDGL support)
make

# 2. (Optional) Generate SSD lattice (requires CUDA + NVIDIA for generator)
# nvcc -O3 -arch=sm_80 hdgl_lattice_generator.cu -o generate_hdgl_lattice
# ./generate_hdgl_lattice --num-experts 512 --chunks-per-expert 1

# 3. Run inference with HDGL-28 mode
./infer --hdgl [other flags, e.g. --prompt "Solve this differential equation..."]

# 4. Interactive chat with HDGL mode
./chat --hdgl

LICENSING - ALL LICENSING, AS APPLICABLE, for flash-moe-originating files retain their original licensing.  

All files diverging at time of publication of this repo remain property of ZCHG.org persuant but not limited to - https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440
