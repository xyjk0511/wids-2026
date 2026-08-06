import pandas as pd, numpy as np, os
from pathlib import Path
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RidgeCV, HuberRegressor
from scipy.special import expit
from src.evaluation import hybrid_score
from src.features import add_engineered_0_97092, get_feature_cols_0_97092

H=[12,24,48,72]; PROB=[f'prob_{h}h' for h in H]
train0=pd.read_csv('train.csv'); test0=pd.read_csv('test.csv'); sample=pd.read_csv('sample_submission.csv')
train=add_engineered_0_97092(train0.copy()); test=add_engineered_0_97092(test0.copy())
# add physics ETA features
for df in [train,test]:
    dist=df['dist_min_ci_0_5h'].clip(lower=1)
    close=df['closing_speed_m_per_h'].clip(lower=0)
    radial=df['radial_growth_rate_m_per_h'].clip(lower=0)
    area=df['area_first_ha'].clip(lower=0)
    radius=np.sqrt(area*10000/np.pi)
    eff=(close+radial).clip(lower=0)
    df['phys_radius_m']=radius
    df['phys_gap_m']=(dist-5000).clip(lower=0)
    df['phys_penetration_m']=(5000-dist).clip(lower=0)
    df['phys_eta_close_h']=np.where(close>1e-3, dist/close, 9999).clip(0,9999)
    df['phys_eta_eff_h']=np.where(eff>1e-3, dist/eff, 9999).clip(0,9999)
    df['phys_eta_edge_h']=np.where(eff>1e-3, (dist-radius).clip(lower=0)/eff, 9999).clip(0,9999)
    df['phys_log_eta_eff']=np.log1p(df['phys_eta_eff_h'])
    df['phys_log_eta_edge']=np.log1p(df['phys_eta_edge_h'])
    df['phys_align_speed']=df['alignment_abs']*close
    df['phys_align_eff']=df['alignment_abs']*eff
    df['phys_threat_flux']=df['alignment_abs']*(radius+df['radial_growth_m'].clip(lower=0))/(dist+100)
    df.replace([np.inf,-np.inf], np.nan, inplace=True); df.fillna(0,inplace=True)

near=train0.dist_min_ci_0_5h<5000; near_test=test0.dist_min_ci_0_5h<5000
features=[c for c in train.columns if c not in ['event_id','event','time_to_hit_hours']]
X=train.loc[near,features]; y=np.log1p(train0.loc[near,'time_to_hit_hours'].values)
Xt=test.loc[:,features]
models=[]
for seed in [11,23,42,77,123,2026]:
    models += [
        ('et', ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.7, random_state=seed, n_jobs=-1)),
        ('rf', RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features='sqrt', random_state=seed, n_jobs=-1)),
        ('gbr', GradientBoostingRegressor(n_estimators=250, learning_rate=0.025, max_depth=2, min_samples_leaf=3, random_state=seed)),
        ('hgb', HistGradientBoostingRegressor(max_iter=180, learning_rate=0.025, max_leaf_nodes=7, l2_regularization=0.1, random_state=seed)),
    ]
models += [('knn5', make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights='distance'))), ('ridge', make_pipeline(StandardScaler(), RidgeCV(alphas=[.1,1,10,100])))]

rkf=RepeatedKFold(n_splits=5,n_repeats=20,random_state=9001)
oof_log=np.zeros(len(X)); cnt=np.zeros(len(X)); test_log_preds=[]
for mi,(name,model) in enumerate(models):
    o=np.zeros(len(X)); c=np.zeros(len(X)); tp=[]
    # fewer folds for slow? okay
    for tr,va in rkf.split(X):
        m=model
        # clone
        import sklearn.base
        m=sklearn.base.clone(model)
        m.fit(X.iloc[tr], y[tr])
        o[va]+=m.predict(X.iloc[va]); c[va]+=1
    m=sklearn.base.clone(model); m.fit(X,y); tp.append(m.predict(Xt))
    oof_log += o/c; cnt+=1
    test_log_preds.append(tp[0])
    print('fit',mi+1,len(models),name)
oof_log/=cnt; test_log=np.mean(test_log_preds,axis=0)
# residual empirical CDF on OOF near
res=y-oof_log

def probs_from(pred_log, temp=1.0, bias=0.0, tail=0.0):
    arr=np.zeros((len(pred_log),4))
    # Smooth residual CDF: mean sigmoid((logh - pred - residual - bias)/temp)
    for j,h in enumerate(H):
        z=(np.log1p(h)-pred_log[:,None]-res[None,:]-bias)/temp
        arr[:,j]=expit(z).mean(axis=1)
    if tail:
        prior=np.array([(train0.loc[near,'time_to_hit_hours']<=h).mean() for h in H])
        arr=(1-tail)*arr+tail*prior
    return np.maximum.accumulate(np.clip(arr,0,1),axis=1)

def full_prob(near_probs, far_mult=0.0, near_scale=1.0):
    d=np.zeros((len(train0),4)); d[near.values]=np.clip(near_probs*near_scale,0,1); d=np.maximum.accumulate(d,axis=1); return {h:d[:,i] for i,h in enumerate(H)}

best=None
for temp in [0.04,0.06,0.08,0.10,0.14,0.18,0.25,0.35,0.5]:
  for bias in [-0.25,-0.15,-0.08,0,0.08,0.15,0.25]:
   for tail in [0,0.05,0.1,0.2]:
    nearp=probs_from(oof_log,temp,bias,tail)
    for ns in [0.92,0.96,1.0,1.04,1.08]:
      pdict=full_prob(nearp, near_scale=ns)
      s,det=hybrid_score(train0.time_to_hit_hours.values, train0.event.values, pdict)
      rec=(s,temp,bias,tail,ns,det)
      if best is None or s>best[0]: best=rec
print('BEST',best[:5],best[5])
# write top variants plus blend with public 850? no oof for public, just output standalone
os.makedirs('submissions',exist_ok=True)
for k,(temp,bias,tail,ns) in enumerate([(best[1],best[2],best[3],best[4]),(0.10,-0.08,0.05,1.0),(0.18,0,0.1,1.0),(0.08,-0.15,0,1.04)]):
    near_test_probs=probs_from(test_log,temp,bias,tail)[near_test.values]
    arr=np.zeros((len(test0),4)); arr[near_test.values]=np.clip(near_test_probs*ns,0,1); arr=np.maximum.accumulate(arr,axis=1)
    sub=pd.DataFrame({'event_id':test0.event_id})
    for i,col in enumerate(PROB): sub[col]=arr[:,i]
    out=f'submissions/submission_exp37_physics_eta_v{k+1}.csv'; sub.to_csv(out,index=False)
    print('wrote',out, 'means',np.round(arr.mean(axis=0),5), 'mono',(np.diff(arr,axis=1)>=-1e-12).all())
# save oof diagnostics
Path('logs').mkdir(exist_ok=True)
Path('logs/exp37_physics_eta.log').write_text('BEST '+repr(best[:5])+' '+repr(best[5])+'\n',encoding='utf-8')
