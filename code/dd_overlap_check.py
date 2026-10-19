"""Step 0 check: (vault,pod,slot) locations where two drives' [first,last] overlap >= 7 days; failures with slot."""
import sys, pandas as pd, numpy as np
for f in sys.argv[1:]:
    d=pd.read_parquet(f,columns=["vault","pod","slot","first","last","fail"])
    nf=int((d.fail==1).sum()); nfs=int(((d.fail==1)&d.slot.notna()).sum())
    x=d.dropna(subset=["vault","pod","slot"]).sort_values(["vault","pod","slot","first"]).reset_index(drop=True)
    key=pd.factorize(pd.MultiIndex.from_frame(x[["vault","pod","slot"]]))[0].astype(np.int64)
    L=x["last"].values.astype("datetime64[D]").astype(np.int64); F=x["first"].values.astype("datetime64[D]").astype(np.int64)
    BIG=10**7; cm=np.maximum.accumulate(L+key*BIG)-key*BIG
    prev=np.r_[0,cm[:-1]]; same=np.r_[False,key[1:]==key[:-1]]
    bad=np.where(same,np.minimum(L,prev)-F,-1)>=7
    nloc=key.max()+1; cnt=np.bincount(key); nbad=len(np.unique(key[bad]))
    print(f"{f}: rows={len(d)} locs={nloc} locs_multi_drive={(cnt>1).sum()} locs_overlap>=7d={nbad} ({nbad/nloc:.4%}) overlapping_rows={int(bad.sum())} fails={nf} fails_with_slot={nfs}",flush=True)
