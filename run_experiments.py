# GenAI acknowledgement: ChatGPT was used for planning, debugging environment
# compatibility, and editing/documentation support. The student reviewed the
# methodology and executed the final experiment locally.

"""SIT307 research reproducibility study. Review/adapt before academic submission."""
import argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

SEED=42
PUBLISHED={'LR':[.8439,.8620,.9090,.8196,.9204], 'NB':[.8439,.8608,.9000,.8250,.9185], 'KNN':[.8585,.8687,.8727,.8648,.9304], 'DT':[.9268,.9289,.8909,.9702,.9702], 'RF':[.9268,.9333,.9545,.9130,.9734], 'XGB':[.9073,.9132,.9090,.9174,.9830], 'Stacking':[.9853,.9861,.9727,1.0000,.9880]}
METRICS=['accuracy','f1','recall','precision','auc']
NUM=['age','trestbps','chol','thalach','oldpeak']
CAT=['sex','cp','fbs','restecg','exang','slope','ca','thal']

def prep():
    # All learned transforms fit inside each training fold; no use of held-out labels.
    return ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUM),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),CAT)],sparse_threshold=0)

def estimators(seed):
    return {'LR':LogisticRegression(max_iter=1500,random_state=seed), 'NB':GaussianNB(), 'KNN':KNeighborsClassifier(n_neighbors=5), 'DT':DecisionTreeClassifier(random_state=seed), 'RF':RandomForestClassifier(n_estimators=100,random_state=seed,n_jobs=1), 'XGB':XGBClassifier(n_estimators=100,max_depth=6,learning_rate=.3,objective='binary:logistic',eval_metric='logloss',random_state=seed,n_jobs=1)}

def models(seed):
    base=estimators(seed)
    res={key:Pipeline([('preprocess',prep()),('model',est)]) for key,est in base.items()}
    # Out-of-fold base-model probabilities for the meta learner, using 5 training-only folds.
    stack=StackingClassifier(estimators=[(key,clone(pipe)) for key,pipe in res.items()],final_estimator=LogisticRegression(max_iter=1500,random_state=seed),cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed),stack_method='predict_proba',passthrough=False,n_jobs=1)
    res['Stacking']=stack
    return res

def scores(y,p,prob):
    return {'accuracy':float(accuracy_score(y,p)),'precision':float(precision_score(y,p,zero_division=0)),'recall':float(recall_score(y,p,zero_division=0)),'f1':float(f1_score(y,p,zero_division=0)),'auc':float(roc_auc_score(y,prob))}

def evaluate(X,y,tr,te,seed,tag,output):
    rows=[]; pred=[]
    for name,est in models(seed).items():
        est.fit(X.iloc[tr],y.iloc[tr]); prob=est.predict_proba(X.iloc[te])[:,1]; p=(prob>=.5).astype(int)
        m=scores(y.iloc[te],p,prob); rows.append({'protocol':tag,'seed':seed,'model':name,'train_n':len(tr),'test_n':len(te),**m})
        if name=='Stacking':
            pred=pd.DataFrame({'row_index':X.index[te],'true':y.iloc[te].values,'probability':prob,'predicted':p});pred.to_csv(output/f'{tag}_stacking_predictions.csv',index=False)
            np.savetxt(output/f'{tag}_stacking_confusion.csv',confusion_matrix(y.iloc[te],p),fmt='%d',delimiter=',',header='TN,FP / FN,TP',comments='# ')
            RocCurveDisplay.from_predictions(y.iloc[te],prob);plt.title(f'{tag}: stacking ROC (test only)');plt.tight_layout();plt.savefig(output/f'{tag}_stacking_roc.png',dpi=160);plt.close()
        print(f'{tag:18} {name:9} accuracy={m["accuracy"]:.4f} AUC={m["auc"]:.4f}',flush=True)
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/heart.csv');ap.add_argument('--output',default='results');ap.add_argument('--seeds',default='42,7,21');a=ap.parse_args()
    output=Path(a.output);output.mkdir(exist_ok=True,parents=True)
    data=pd.read_csv(a.data);expected=NUM+CAT+['target'];assert set(data)==set(expected),(set(data),set(expected))
    assert not data.isna().any().any(), 'Inspect missingness and justify imputation';assert set(data.target.unique())=={0,1}
    X=data.drop(columns='target');y=data.target.astype(int)
    # Group by all 13 feature values (not target). Equal feature records must share a split.
    groups=pd.factorize(pd.MultiIndex.from_frame(X))[0]
    unique=data.drop_duplicates();audit={'records':len(data),'unique_full_rows':len(unique),'duplicate_full_rows':int(data.duplicated().sum()),'distinct_feature_vectors':int(len(np.unique(groups))),'conflicting_feature_labels':int(data.groupby(list(X.columns)).target.nunique().gt(1).sum()),'target_counts':{str(k):int(v) for k,v in y.value_counts().items()}}
    (output/'data_audit.json').write_text(json.dumps(audit,indent=2));print('DATA AUDIT',audit,flush=True)
    rows=[];tr,te=train_test_split(np.arange(len(X)),test_size=.2,random_state=SEED,stratify=y)
    overlap=len(set(groups[tr])&set(groups[te]));audit['random_split_test_rows_with_train_duplicate']=int(np.isin(groups[te],groups[tr]).sum());audit['random_split_shared_feature_groups']=int(overlap)
    (output/'data_audit.json').write_text(json.dumps(audit,indent=2))
    rows+=evaluate(X,y,tr,te,SEED,'random_80_20',output)
    # Deduplicate first; evaluate unseen feature vectors, preserving 13 features.
    Xu=unique.drop(columns='target');yu=unique.target.astype(int)
    seeds=[int(s) for s in a.seeds.split(',')]
    for seed in seeds:
        tr,te=train_test_split(np.arange(len(Xu)),test_size=.2,random_state=seed,stratify=yu)
        rows+=evaluate(Xu,yu,tr,te,seed,f'unique_seed_{seed}',output)
    results=pd.DataFrame(rows);results.to_csv(output/'all_metrics.csv',index=False)
    published=pd.DataFrame([{'model':k,**dict(zip(METRICS,values))} for k,values in PUBLISHED.items()]);published.to_csv(output/'published_table11_transcribed.csv',index=False)
    baseline=results.query('protocol == "random_80_20"').drop(columns=['protocol','seed','train_n','test_n']).merge(published,on='model',suffixes=('_reproduced','_paper'));baseline.to_csv(output/'paper_comparison.csv',index=False)
    summary=results.query('protocol != "random_80_20"').groupby('model')[METRICS].agg(['mean','std']);summary.to_csv(output/'unique_repeated_summary.csv')
    fig,ax=plt.subplots(figsize=(10,5));order=list(PUBLISHED);v=baseline.set_index('model').loc[order];x=np.arange(len(order));w=.3
    ax.bar(x-w/2,v.accuracy_paper,w,label='Paper reported');ax.bar(x+w/2,v.accuracy_reproduced,w,label='Our random split');ax.set_xticks(x,order);ax.set_ylim(0,1.05);ax.set_ylabel('Held-out accuracy');ax.legend();fig.tight_layout();fig.savefig(output/'accuracy_comparison.png',dpi=160);plt.close(fig)
    print('FINISHED',output.resolve(),flush=True)
if __name__=='__main__':main()
