
```angular2html
python train.py ENNS2V --train-set toydata/piece-of-tox21-train.csv.gz --valid-set toydata/piece-of-tox21-valid.csv.gz --test-set toydata/piece-of-tox21-test.csv.gz --loss MaskedMultiTaskCrossEntropy --score roc-auc --s2v-lstm-computations 9 --out-hidden-dim 150 --logging more --epochs 20 --learn-rate 0.0001

```




```angular2html
python train.py GGNN --train-set data/ESOL_train.csv.gz --valid-set data/ESOL_valid.csv.gz --test-set data/ESOL_test.csv.gz --score RMSE --loss MSE --logging less --epochs 1200 --learn-rate 1.176e-5
```

batch size 50, learn rate 1.176e-5 and 1200 epochs is good for ESOL

```angular2html
nohup python train.py GGNN --train-set data/ESOL_train.csv.gz --valid-set data/ESOL_valid.csv.gz --test-set data/ESOL_test.csv.gz --score RMSE --loss MSE --logging less --epochs 2000 --learn-rate 1.176e-5 > outputs/GGNN_ESOL_predicted_log_solubility.log 2>&1 &
```