"""
check_layers.py

Prints the layer count (and a few other facts) of a GGUF model by reading only
the file header. No weights are loaded, so it runs in a second.

Usage:
  python3 check_layers.py                      # uses the default model below
  python3 check_layers.py /path/to/model.gguf  # or pass any GGUF file
"""

import os
import sys

from llama_cpp import Llama

DEFAULT_MODEL = os.path.expanduser(
    "~/models/nemotron-3.5-bolt-topo-governed-Q4_K_M.gguf"
)


def main() -> int:
    path = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return 1

    llm = Llama(model_path=path, vocab_only=True, verbose=False)
    meta = llm.metadata
    arch = meta.get("general.architecture")

    print(f"File         : {os.path.basename(path)}")
    print(f"Architecture : {arch}")

    layers = meta.get(f"{arch}.block_count")
    if layers is None:
        print("Layers       : not found in the file header")
        print("\nAll keys ending in 'count' or 'length':")
        for k, v in meta.items():
            if k.endswith(("count", "length")):
                print(f"  {k} = {v}")
        return 1

    print(f"Layers       : {layers}")
    print(f"  (llama.cpp can offload up to {int(layers) + 1} when the output layer is counted)")

    for label, key in [
        ("Context len  ", f"{arch}.context_length"),
        ("Embedding dim", f"{arch}.embedding_length"),
        ("Attn heads   ", f"{arch}.attention.head_count"),
    ]:
        if key in meta:
            print(f"{label} : {meta[key]}")

    llm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())