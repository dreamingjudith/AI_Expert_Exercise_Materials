# Transformers and Sequences

**본 실습은 Pycharm에 맞게 설정되어 있습니다.**

```
$ python train.py -data_pkl=m30k_deen_shr.pkl -log=m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model=trained -b=128 -warmup=2000 -epoch=30
```

```
$ python translate.py -model=trained.chkpt -data_pkl=m30k_deen_shr.pkl -output=result.txt
```

