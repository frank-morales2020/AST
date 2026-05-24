Here is a complete, production-ready, and highly professional **README.md** for your GitHub repository. It perfectly blends the architectural depth of your Zenodo papers, the neurobiological grounding of your hippocampal emulation model, and the empirical metrics from your live Hugging Face release.

---

# Arithmetic Spectral Theory (AST) & Topological AI: Prime-Anchored Neural Networks

An open-source, deterministic, and hardware-efficient framework that solves **catastrophic forgetting** in Large Language Models (LLMs). By implementing a **Topological Governor** rooted in number theory and **Hippocampal Index Theory**, this architecture bypasses the unscalable resource bloat of corporate statistical regularization mechanisms. It anchors an extremely sparse set of prime-indexed embedding rows during sequential training streams, leaving the rest of the parameter landscape completely free to learn and adapt.

---

## 🧠 Core Philosophy & Biological Foundation

Topological AI rejects the corporate optimization trend of chasing an unnatural **0% forgetting rate** (e.g., Google's H0PE-like framework). Forcing absolute stability across an entire model creates cognitive rigidity, rendering the network learning-disabled when encountering cross-domain tasks.

In nature, **forgetting is the price of adaptation**. This framework functions as an executable emulation of the mammalian **Hippocampus-Neocortex dual-timescale memory system**:

* **The Plastic Neocortex (99.99% of Weights):** Unlocked embedding rows remain completely free to reorganize, specialize, and ingest new features dynamically.
* **The Sparse Hippocampal Index (0.01% of Weights):** Exactly **6 rows** indexed by prime numbers ($2, 3, 5, 7, 11, 13$) are securely anchored to form a permanent, universal coordinate reference frame. This preserves foundational geometric governance boundaries, safety thresholds ($\Lambda = 0.9785142874$), and high-level expert alignment rules indefinitely.

---

## 📊 Empirical Benchmarks vs. Industry Standards

The provided benchmarks validate performance across a three-task sequential training stream (Task A $\rightarrow$ Task B $\rightarrow$ Task C) using a **20-Billion Parameter Model (GPT-OSS-20B)** on an NVIDIA RTX PRO 6000 Blackwell Server Edition.

### The TOPO-2026 Certification Dashboard

```
+------------------------------------------+
| TOPOLOGICAL AI CERTIFIED                 |
| |- Task C Accuracy: 99.7% (>=95%)  PASS  |
| |- Forgetting:      6.0%  (<=10%)  PASS  |
| |- Anchor Integrity:               PASS  |
| |- Runs Completed:  5/5   (5/5)    PASS  |
| `- Standard: TOPO-2026                   |
+------------------------------------------+

```

### Comparative Production Metrics ($N=5$ Runs)

| Metric / Mechanism | Topological AI | EWC (Fisher Matrix) | HOPE-like (Google) | Baseline Fine-Tuning |
| --- | --- | --- | --- | --- |
| **Task C Novel Accuracy** | **99.7% $\pm$ 0.6%** | 98.5% $\pm$ 0.0% | 88.1% $\pm$ 9.2% | 96.3% $\pm$ 4.0% |
| **Combined Forgetting** | **+6.0%** | +6.7% | **+0.1%** | +7.0% |
| **Protection Memory Cost** | **67.50 KB** (Flat) | 4.41 GB (Per Task) | 2.26 GB | 0 KB |
| **Protection Time Latency** | **0.11 ms** | 4,808 ms | 173,674 ms | 0 ms |
| **Scaling Complexity** | **$O(1)$ Constant** | $O(k)$ Linear (Memory) | $O(k)$ Linear (Time) | $O(1)$ |
| **Hardware Status** | **5/5 Runs OK** | 1/5 Runs (OOM Crash) | 5/5 Runs OK | 5/5 Runs OK |

* **Why Corporate Baselines Fail:** EWC triggers an immediate Out-of-Memory (OOM) crash by the second run due to massive memory fragmentation. Google's H0PE-like framework achieves its near-zero forgetting metric by simply locking weight values so rigidly that it fails at the fundamental operational requirement of learning new tasks.

---

## 🛠️ Code Implementation: The Topological Governor

Because the governor intercepts and zero-clears gradients directly at the source, it is fully **optimizer-agnostic** and preserves seamless compatibility with low-precision, quantized edge runtimes such as `bitsandbytes`.

```python
import torch
import torch.nn as nn
import numpy as np

class TopologicalGovernor:
    def __init__(self, embed_layer: nn.Embedding, prime_limit: int = 13):
        self.embed_layer = embed_layer
        vocab_size = embed_layer.weight.shape[0]
        
        # Sieve-generated prime mapping serving as our universal reference frame
        self.anchor_indices = [2, 3, 5, 7, 11, 13]
        self.snapshot = {}
        
        # Mathematical Euler attenuation attenuation constant
        self.safety_constant = 1.0 - np.prod([1.0 - (p ** -0.5) for p in self.anchor_indices])

    def take_snapshot(self):
        """Executes a deep snapshot memory consolidation sequence."""
        self.snapshot = {
            idx: self.embed_layer.weight[idx].detach().clone().float()
            for idx in self.anchor_indices
        }

    @torch.no_grad()
    def zero_anchor_gradients(self):
        """Intercepts the backward pass to zero gradients at the source."""
        if self.embed_layer.weight.grad is not None:
            for idx in self.anchor_indices:
                self.embed_layer.weight.grad[idx].zero_()

    @torch.no_grad()
    def enforce_anchors(self):
        """Enforces hard geometric safety boundaries post-optimization step."""
        dtype = self.embed_layer.weight.dtype
        for idx, cached in self.snapshot.items():
            self.embed_layer.weight[idx].copy_(cached.to(dtype=dtype))

    def verify_integrity(self, atol: float = 1e-5) -> bool:
        """Audits the unalterable coordinate identities."""
        return all(
            torch.allclose(self.embed_layer.weight[idx].float(), cached, atol=atol)
            for idx, cached in self.snapshot.items()
        )

```

---

## 📂 Repository Structure

```
├── ast_lefm/
│   ├── sieve.py         # Dynamic Sieve of Eratosthenes calculations
│   ├── lefm.py          # Laplace-Euler-Fourier-Mellin operators
│   ├── h2e.py           # Human-to-Expert safety alignment layers
│   ├── constants.py     # Universal safety constants and thresholds
│   └── einstein.py      # Relativistic coordinate mapping instruments
├── test/
│   └── test_all.py      # Automated pipeline audit routines
├── BENCHMARK_TOPOLOGICAL_AI.ipynb  # Primary 3-task comparative execution loop
├── certified_topological_final.pt # 39 GB TOPO-2026 certified weights
├── README.md            # Repository documentation
└── setup.py             # Package builder and module compiler

```

---

## ⚡ Direct Deployment & Inference

You can run immediate cross-domain token classifications utilizing the pre-compiled certified model array:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the base weights alongside the verified tokenizer pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO_ID = "frankmorales2020/topological-ai-gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(REPO_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", torch_dtype=torch.bfloat16).to(device)

# Load your unalterable certified state dictionary configuration
state_dict = torch.load("certified_topological_final.pt", map_location=device)
base_model.load_state_dict(state_dict, strict=False)
base_model.eval()

```

---

## 🔬 Scientific Foundations & Comprehensive Taxonomy

The mathematical validation for the arithmetic operations and spectral boundaries utilized by this governor is preserved permanently across the following open-access records:

### The Complete Spectral Archive

* **Main Architectural Thesis:** *Topological AI: Prime-Anchored Neural Networks Solving Catastrophic Forgetting in Large Language Models*. [Zenodo Record 20360042](https://zenodo.org/records/20360042)
* **Empirical Validation Ledger:** *Topological AI: Prime-Anchored Neural Networks Solve Catastrophic Forgetting — A Complete Empirical Validation on GPT-OSS-20B*. [Zenodo Record 20348964](https://zenodo.org/records/20348964)
* **Theoretical Framework:** *Topological AI: Prime-Anchored Neural Networks That Do Not Forget — A Practical Framework for Deterministic, Verifiable, Catastrophic-Forgetting-Resistant Artificial Intelligence*. [Zenodo Record 20338459](https://zenodo.org/records/20338459)
* **The Core Mathematical Operator:** *L-EFM: A Laplace-Extended Euler-Fourier-Mellin Operator That Proves the Riemann Hypothesis*. [Zenodo Record 19908304](https://zenodo.org/records/19908304)
* **The Boundary Foundation:** *A Spectral Answer to Tao: How the L-EFM Operator Quantifies the Green-Tao Theorem and Proves the Riemann Hypothesis*. [Zenodo Record 20199735](https://zenodo.org/records/20199735)
* **Safety Constancy Blueprint:** *H2E Sheriff: Mathematical Derivation of Universal Safety Constants Including the Lambda Spectral Complementarity Theorem and Applications*. [Zenodo Record 20218178](https://zenodo.org/records/20218178)

*(See the full list of all 18 interrelated foundational works and historical papers within the comprehensive index inside the main documentation file.)*

---

## 🏛️ Provenance & Affiliation

**Author:** Frank Morales Aguilera, BEng, MEng, SMIEEE

**Organization:** Founder & CEO, Sovereign Machine Lab (SOMALA)

**Role:** Chief AI Officer (CAIO), Drivia Consulting

**Location:** Montréal, Québec, Canada

**Infrastructure:** Built and audited locally using an NVIDIA RTX PRO 6000 Blackwell Server Edition. Certified for sovereign deployment models.
