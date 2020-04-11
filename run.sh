srun --gres=gpu:1 python -u main.py train --num_epoch=20 --use_gpu=True > train.log &
