# HGNN_EHR



### Train

```shell
python main.py train --use_gpu=True --num_epoch=20
```



### Evaluate

```shell
python eval.py run --use_gpu=True --ckpt=ckpt/checkpoint.pt --top_k=3
```

