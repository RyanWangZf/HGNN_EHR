#srun --gres=gpu:1 python -u main.py train --num_epoch=50 --lr=0.01 --batch_size=1024 --hard_ratio=0 --weight_decay=1e-4 --use_gpu=True > train.log &
#python -u main.py train --num_epoch=50 --lr=0.01 --batch_size=1024 --hard_ratio=0 --weight_decay=1e-4 --use_gpu=False
srun --gres=gpu:1 python -u main_mf.py train --lr=0.05 --num_epoch=50 --batch_size=128 --use_gpu=True > train_mf.log &
#srun --gres=gpu:1 python -u main_neumf.py train --lr=0.01 --num_epoch=50 --batch_size=128 --use_gpu=True > train_neumf.log &
#srun --gres=gpu:1 python -u main_textcnn.py train --num_epoch=50 --lr=0.01 --use_gpu=True --batch_size=32 > train_tcnn.log &
