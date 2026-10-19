import numpy as np, pandas as pd, sys
rng=np.random.default_rng(1); planted=int(sys.argv[2]); rows=[]
T=180; h=0.00006
for v in range(30):
  for p in range(20):
    for s in range(60):
      rows.append(dict(serial_number=f"{v}-{p}-{s}",model="M",vault=float(v),pod=float(p),slot=float(s),n_pods_seen=1,first=0,last=T-1,fd=np.nan,poh0=1000.0))
d=pd.DataFrame(rows)
fd=np.where(rng.random(len(d))<h*T, rng.integers(0,T,len(d)), np.nan).astype(float)
if planted:
  key=d.vault.values*10000+d.pod.values*100+(d.slot.values//5)
  for j in np.where(~np.isnan(fd))[0]:
    m=(key==key[j])&np.isnan(fd); m[j]=False
    hit=m&(rng.random(len(d))<0.04)
    fd[hit]=fd[j]+rng.integers(1,31,hit.sum())
fd[fd>=T]=np.nan
p0=pd.Timestamp("2024-01-01")
d["first"]=p0; d["last"]=p0+pd.Timedelta(days=T-1); d["fail"]=(~np.isnan(fd)).astype(int)
d["fdate"]=[p0+pd.Timedelta(days=x) if not np.isnan(x) else pd.NaT for x in fd]
d=d.drop(columns="fd"); d["period_start"]="2024-01-01"; d["period_end"]=str((p0+pd.Timedelta(days=T-1)).date()); d["n_files"]=T
d.to_parquet(sys.argv[1],index=False); print(d.fail.sum())
