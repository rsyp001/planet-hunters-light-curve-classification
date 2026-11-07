# Planet Hunters: light-curve classification (ML Challenge Hack4Hack 2026, Round 1)

Classify 25,000 stars (27 days of 30-minute brightness readings each) into 8 classes:
planet, eclipsing binary, background binary, starspots, pulsating star, flare star,
instrument glitch, quiet star. Metric: **macro-F1**. The test set comes from a different
observing sector than the training set.

## Results
| | macro-F1 |
|---|---|
| 5-fold CV (grouped by camera/CCD), argmax | 0.7045 |
| 5-fold CV, tuned class multipliers | 0.7156 |
| Public leaderboard (30% of test) | 0.6678 |
| **Private leaderboard (hidden 70%)** | **0.6508** |

The ~0.065 drop from CV to the private score is attributed to train/test sector shift
(more scattered light and different thruster-dump timing in the test sector).

## Approach
1. **Cleaning:** divide each star by its median; subtract the per-CCD median curve
   (computed within each sector) from flux and centroids; 2-day running-median high-pass.
2. **Features (~100):** robust statistics; FFT band powers and peaks; autocorrelation;
   flare runs and decay; jump/orbit/dump glitch features; BLS transit search with
   dip vetting (odd/even depth, secondary eclipse, U vs V shape, per-transit consistency);
   centroid shift during the dip; stellar properties.
3. **Model:** LightGBM multiclass, class-weighted, early stopping,
   StratifiedGroupKFold by camera/CCD.
4. **Drift handling:** features that separate train from test (univariate AUC) are
   rank-normalized within each sector; dump-timing features dropped.
5. **Decision rule:** per-class probability multipliers tuned on out-of-fold predictions
   to maximize macro-F1 (tuned on the same OOF predictions, so the gain is slightly optimistic).

## Run
Download the competition data into a folder, then:
```
pip install -r requirements.txt
export DATA=/path/to/data
python features.py test
python features.py train
python train.py          # writes submission.csv
```
Feature extraction takes ~16 minutes on one CPU core. The competition data is not
included in this repo.

## Limitations and next steps
- Weakest classes: starspots (F1 0.57) and planet (0.61), mostly confused with quiet stars.
- Test predictions contain more starspots than the train class mix, a likely residue of the sector shift.
- Ideas not yet tried: finer, noise-weighted BLS grid; 1D CNN on folded/detrended curves
  blended with the GBM; sector-aware domain adaptation.
