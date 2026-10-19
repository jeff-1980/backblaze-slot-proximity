"""Write copies of the per-drive tables under the original server key of the frozen plan, (vault_id, pod_id), by
setting vault = vault_raw. Added 2026-10-19 as the entry point for the original-key rows of Supplementary Table 2.
USAGE  python dd_original_key.py <out_dir> <k.parquet> [...]
"""
import sys, os
import pandas as pd

os.makedirs(sys.argv[1], exist_ok=True)
for p in sys.argv[2:]:
    d = pd.read_parquet(p)
    d["vault"] = d["vault_raw"]
    d.to_parquet(os.path.join(sys.argv[1], os.path.basename(p)), index=False)
    print(p, len(d))
