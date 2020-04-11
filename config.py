# -*- coding: utf-8 -*-    


def default_config():
    model_param = {
            "symp_embedding_dim": 64,
            "dise_embedding_dim": 64,
            "layer_size_dsd": [64,64],
            "layer_size_usu": [64,64,64],
            "dropout_ratio": 0.1,
            "lr":1e-3,
            "weight_decay":1e-3,
            "batch_size":1024,
            "num_epoch":100,
            "early_stop":5,
            "use_gpu":True,
        }

    return model_param