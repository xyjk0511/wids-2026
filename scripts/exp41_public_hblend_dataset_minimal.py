import pandas as pd, numpy as np, os
from pathlib import Path
DATA='external_kernels/data-submission/'
OUT='submissions/submission_exp41_hblend_dataset.csv'
solutions=['0.97124','0.97167','0.97169','9724']
weights=np.array([0.1,0.1,0.1,0.7])
subwts=np.array([-0.07,-0.03,-0.01,0.11])
asc_w, desc_w = 0.30, 0.70
probs=['prob_12h','prob_24h','prob_48h','prob_72h']

def hblend_col(dfs, col):
    ids=dfs[0]['event_id']
    mat=np.column_stack([df[col].to_numpy(float) for df in dfs])
    # desc: highest value gets rank 0 -> subwts[0]; asc: lowest value gets rank 0
    order_desc=np.argsort(-mat, axis=1)
    order_asc=np.argsort(mat, axis=1)
    rank_desc=np.empty_like(order_desc); rank_asc=np.empty_like(order_asc)
    for i in range(mat.shape[0]):
        rank_desc[i, order_desc[i]]=np.arange(mat.shape[1])
        rank_asc[i, order_asc[i]]=np.arange(mat.shape[1])
    desc=np.sum(mat*(weights[None,:]+subwts[rank_desc]),axis=1)
    asc=np.sum(mat*(weights[None,:]+subwts[rank_asc]),axis=1)
    return asc_w*asc + desc_w*desc

dfs=[pd.read_csv(os.path.join(DATA, s+'.csv')) for s in solutions]
out=dfs[0][['event_id']].copy()
for col in probs:
    out[col]=hblend_col(dfs,col)
arr=np.clip(out[probs].to_numpy(float),0,1)
arr=np.maximum.accumulate(arr,axis=1)
out[probs]=arr
Path('submissions').mkdir(exist_ok=True)
out.to_csv(OUT,index=False)
print('wrote',OUT,out.shape,dict(zip(probs,np.round(arr.mean(axis=0),5))))
