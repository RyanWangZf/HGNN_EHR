# -*- coding: utf-8 -*-    


def default_config():
    model_param = {
            "symp_embedding_dim": 16,
            "dise_embedding_dim": 16,
            "layer_size_dsd": [16,16],
            "layer_size_usu": [16,16,16],
            "dropout_ratio": 0.1,
            "lr":1e-3,
            "weight_decay":1e-4,
            "batch_size":128,
            "num_epoch":50,
            "early_stop":5,
            "use_gpu":True,
        }

    return model_param