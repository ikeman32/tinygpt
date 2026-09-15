# Tiny GPT

A 30-million-parameter transformer model trained from scratch and aligned via Supervised Fine-Tuning (SFT) entirely on consumer laptop CPU hardware. Based on the architecture inspired by [tinynanogpt](https://github.com/garyexplains/examples/tree/master/tinynanogpt_chartok).

The goal of this project is to explore how much practical instruction-following capability, domain specificity (Linux CLI, C, Python), and conversational stability can be extracted from an ultra-compact parameter footprint without dedicated GPU acceleration.

---

### Hardware & Environment

```text
System:
  Kernel: 7.0.0-31-generic arch: x86_64 bits: 64 compiler: gcc v: 13.3.0
  Desktop: Cinnamon v: 6.6.9 Distro: Linux Mint 22.3 Zena
    base: Ubuntu 24.04 noble
Machine:
  Type: Laptop System: HP product: HP Laptop 15-fc0xxx v: N/A
  Mobo: HP model: 8DC7 v: 53.51 UEFI: AMI v: F.17
CPU:
  Info: 8-core model: AMD Ryzen 7 7730U with Radeon Graphics (16 threads, Zen 3)
```
---

### Getting Started

Clone the repository and set up the environment using [uv](https://github.com/astral-sh/uv):

```bash
git clone <REPO_URL>
cd <REPO_NAME>
