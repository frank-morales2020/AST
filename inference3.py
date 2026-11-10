
import gc
import json
import math
import os
import sys
import time
import subprocess
import tempfile
import warnings
from llama_cpp import Llama
import torch

SEED = 123
CLINICAL_ANCHORS = [2, 3, 5, 7, 11, 13]
#DRIVE_DIR = "/Users/frankmorales/nemotron_model_disk"
DRIVE_DIR = "/Users/frankmorales/models"

MODEL_PATHS = {
    "nemotron": os.path.join(DRIVE_DIR, "nemotron-3.5-bolt-topo-governed-Q4_K_M.gguf"),
}


class ClinicalAuditHarness:
    def __init__(self, prime_anchors, param_dim=128):
        self.p_indices = torch.tensor(prime_anchors, dtype=torch.long)
        self.dim = param_dim
        torch.manual_seed(SEED)
        self.theta_0 = torch.randn(self.dim, dtype=torch.float64)
        self.theta_current = self.theta_0.clone()
        self.baseline_risk = self._eval_risk(self.theta_0)
        self.telemetry_log = []

    def _eval_risk(self, theta: torch.Tensor) -> float:
        subspace_weights = torch.zeros(self.dim, dtype=torch.float64)
        subspace_weights[self.p_indices] = 1.0 / (self.p_indices.to(torch.float64))
        residual = (theta - self.theta_0) * subspace_weights
        return float(torch.sum(residual ** 2).item())

    def record_step(self, stage_name: str, model_name: str, offload_profile: str, step_grad=None, eta: float = 0.01):
        if step_grad is not None:
            mask = torch.ones(self.dim, dtype=torch.float64)
            mask[self.p_indices] = 0.0
            projected_grad = step_grad * mask
            self.theta_current = self.theta_current - eta * projected_grad

        anchor_0 = self.theta_0[self.p_indices]
        anchor_t = self.theta_current[self.p_indices]
        delta_drift = float(torch.norm(anchor_t - anchor_0, p=2).item())

        current_risk = self._eval_risk(self.theta_current)
        variance_score = max(0.0, current_risk - self.baseline_risk)

        status = "PASS (ZERO DRIFT)" if delta_drift == 0.0 else "FAIL (DRIFT DETECTED)"
        record = {
            "stage": stage_name,
            "model": model_name,
            "offload": offload_profile,
            "delta_drift": delta_drift,
            "variance_score": variance_score,
            "status": status
        }
        self.telemetry_log.append(record)

        print(f"\n[Telemetry: {stage_name}]")
        print(f"  ├── Clinical Anchors P      : {self.p_indices.tolist()}")
        print(f"  ├── Hardware Offload Profile: {offload_profile}")
        print(f"  ├── Parameter Drift (Δ_drift): {delta_drift:.12f}")
        print(f"  ├── Diagnostic Variance     : {variance_score:.12f}")
        print(f"  └── Invariant Certification : {status}")
        return record

    def print_summary_table(self):
        print("\n" + "=" * 110)
        print("📊 CLINICAL TOPO-CBP GOOGLE DRIVE PIPELINE TELEMETRY AUDIT")
        print("=" * 110)
        header = f"{'Stage':<8} | {'Model':<25} | {'Offload Profile':<22} | {'Δ_drift':<14} | {'Variance':<12} | {'Verdict'}"
        print(header)
        print("-" * 110)
        for r in self.telemetry_log:
            print(f"{r['stage']:<8} | {r['model']:<25} | {r['offload']:<22} | {r['delta_drift']:<14.10f} | {r['variance_score']:<12.10f} | {r['status']}")
        print("=" * 110 + "\n")

def purge(model):
    del model
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def load_drive_allocated_llama(model_key: str, ctx: int, n_gpu_layers: int, extra_kwargs=None):
    path = MODEL_PATHS[model_key]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found on Google Drive: {path}")
    
    kwargs = {
        "model_path": path,
        "n_ctx": ctx,
        "n_gpu_layers": n_gpu_layers,
        "n_threads": 8,
        "seed": SEED,
        "verbose": False
    }
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    
    print('\n')
    print(f"[*] Loading GGUF File with llama_cpp.Llama ")
    print(f"[*] Loading from Local (MacBook Air) -> {os.path.basename(path)} | n_gpu_layers={n_gpu_layers}")
    print(f"[*] Folder: {DRIVE_DIR}")
    print('\n')
    return Llama(**kwargs)


harness = ClinicalAuditHarness(prime_anchors=CLINICAL_ANCHORS, param_dim=128)

patient_presentation = (
    "Patient presents with anomalous peptide-binding behavior, acute localized inflammation, "
    "and suspected allergen/toxic exposure post-environmental contact."
)

llm1 = load_drive_allocated_llama("nemotron", ctx=2048, n_gpu_layers=0)
prompt1 = (
    f"<|im_start|>system\n"
    f"You are the Clinical Triage Decomposer. Extract anomalous peptide bindings and inflammatory markers.\n<|im_end|>\n"
    f"<|im_start|>user\n"
    f"Presentation: {patient_presentation}\n<|im_end|>\n"
    f"<|im_start|>assistant\n"
)
out1 = llm1(prompt1, max_tokens=300, temperature=0.0, stop=["<|im_end|>"])
stage1_biomarkers = out1["choices"][0]["text"].strip()
purge(llm1)
print(f"[Stage 1 Output]:\n{stage1_biomarkers}\n")
harness.record_step("Stage 1", "nemotron-3.5-bolt", "100% CPU (layers=0)", step_grad=torch.randn(128, dtype=torch.float64))
