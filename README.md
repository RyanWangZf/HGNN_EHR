# HGNN_EHR

### Processing

```shell
cd dataset
python ehr_large.py
```



### Train

```shell
python main.py train --use_gpu=True --num_epoch=20 --lr=0.01 --batch_size=1024 --hard_ratio=0 --weight_decat=1e-4 --w2v="./ckpt/w2v"
```



### Evaluate

```shell
python eval.py run --use_gpu=True --ckpt=ckpt/checkpoint.pt --top_k=3
```

