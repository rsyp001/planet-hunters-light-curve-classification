import numpy as np, pandas as pd, lightgbm as lgb, sys, json
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score

import os
D = os.environ.get('DATA', '.')
tr = pd.read_csv(f'{D}/train_stars.csv'); te = pd.read_csv(f'{D}/test_stars.csv')
Xtr = pd.read_parquet(f'{D}/feat_train.parquet'); Xte = pd.read_parquet(f'{D}/feat_test.parquet')
classes = sorted(tr.label.unique()); cmap = {c: i for i, c in enumerate(classes)}
y = tr.label.map(cmap).values
groups = (tr.camera * 10 + tr.ccd).values

# ---- drift screen: univariate AUC train-vs-test, drop features that separate sectors too much
drop = []; drift = []
for c in Xtr.columns:
    a = pd.concat([Xtr[c], Xte[c]]).fillna(-999).values
    lab = np.r_[np.zeros(len(Xtr)), np.ones(len(Xte))]
    auc = roc_auc_score(lab, a)
    if abs(auc - 0.5) > 0.05:
        drift.append(c)
    if c in ('frac_tr_near_dump', 'orbit_med_diff', 'dump_resp', 'dump_absresp', 'frac_in_near_edge'):
        drop.append((c, round(auc, 3)))
print('drifted features dropped:', drop)
keep = [c for c in Xtr.columns if c not in dict(drop)]
Xtr, Xte = Xtr[keep].copy(), Xte[keep].copy()
for c in drift:
    if c in keep:
        Xtr[c] = Xtr[c].rank(pct=True); Xte[c] = Xte[c].rank(pct=True)
print('rank-normalized:', [c for c in drift if c in keep])

params = dict(objective='multiclass', num_class=8, learning_rate=0.04, num_leaves=31, min_child_samples=20,
              feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, n_jobs=1)
cw = len(y) / (8 * np.bincount(y)); w = cw[y]

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
oof = np.zeros((len(y), 8)); pte = np.zeros((len(Xte), 8)); imp = np.zeros(len(keep))
for k, (a, b) in enumerate(skf.split(Xtr, y, groups)):
    dtr = lgb.Dataset(Xtr.iloc[a], y[a], weight=w[a]); dva = lgb.Dataset(Xtr.iloc[b], y[b], weight=w[b])
    m = lgb.train(params, dtr, 1500, valid_sets=[dva], callbacks=[lgb.early_stopping(60, verbose=False)])
    oof[b] = m.predict(Xtr.iloc[b], num_iteration=m.best_iteration)
    pte += m.predict(Xte, num_iteration=m.best_iteration) / 5
    imp += m.feature_importance('gain')
    print(f'fold {k} iters {m.best_iteration} macroF1 {f1_score(y[b], oof[b].argmax(1), average="macro"):.4f}', flush=True)
print('OOF macro-F1 (argmax):', round(f1_score(y, oof.argmax(1), average='macro'), 4))
print(pd.DataFrame(confusion_matrix(y, oof.argmax(1)), index=classes, columns=[c[:6] for c in classes]))
print('per-class F1:', dict(zip(classes, f1_score(y, oof.argmax(1), average=None).round(3))))

# ---- per-class multipliers (coordinate ascent) for macro-F1
wts = np.ones(8)
best = f1_score(y, (oof * wts).argmax(1), average='macro')
for _ in range(3):
    for c in range(8):
        for s in (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0):
            w2 = wts.copy(); w2[c] = wts[c] * s
            sc = f1_score(y, (oof * w2).argmax(1), average='macro')
            if sc > best + 1e-5:
                best, wts = sc, w2
print('tuned OOF macro-F1:', round(best, 4), 'weights', wts.round(2))

imp = pd.Series(imp, index=keep).sort_values(ascending=False)
print(imp.head(25))
np.save(f'{D}/oof2.npy', oof); np.save(f'{D}/pte2.npy', pte); np.save(f'{D}/wts2.npy', wts)
sub = pd.DataFrame({'id': te.id, 'label': [classes[i] for i in (pte * wts).argmax(1)]})
sub.to_csv(f'{D}/submission.csv', index=False)
print(sub.label.value_counts())
