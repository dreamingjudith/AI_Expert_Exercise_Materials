# Transformers and Sequences

4주 3일차 **Transformers and Sequences** 실습 파일입니다. 본래 Pycharm에서 실행되도록 작성된 코드이나 커맨드 환경에서도 스크래치에서부터 동작하게끔 일부 수정하였습니다.

## Using transformer models in `seq2seq_transformer` folder

### Preprocessing

Download model with spacy, but I don't know the exact meaning

```
$ python -m spacy download en
$ python -m spacy download de
```

Run preprocess.py to create train dataset

```
$ python preprocess.py -lang_src de -lang_trg en -share_vocab -save_data m30k_deen_shr.pkl
```

### Train

```
$ python train.py -data_pkl=m30k_deen_shr.pkl -log=m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model=trained -b=128 -warmup=2000 -epoch=30
```

### Inference

```
$ python translate.py -model=trained.chkpt -data_pkl=m30k_deen_shr.pkl -output=result.txt
```

