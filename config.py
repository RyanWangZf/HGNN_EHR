# -*- coding: utf-8 -*-    


def default_config():
    model_param = {
            "symp_embedding_dim": 64,
            "dise_embedding_dim": 64,
            "layer_size_dsd": [64,64],
            "layer_size_usu": [64,64,64],
            "dropout_ratio": 0.2,
            "lr":0.01,
            "weight_decay":1e-3,
            "batch_size":1024,
            "num_epoch":100,
            "early_stop":5,
            "hard_ratio":0.2,
            "use_gpu":True,
            "dataset":"EHR",
            "w2v":None,
        }

    return model_param