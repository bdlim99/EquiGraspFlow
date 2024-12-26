from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop


def get_optimizer(cfg, model_params):
    name = cfg.pop('name')

    optimizer_class = get_optimizer_class(name)

    optimizer = optimizer_class(model_params, **cfg)

    return optimizer


def get_optimizer_class(name):
    try:
        return {
            'sgd': SGD,
            'adam': Adam,
            'asgd': ASGD,
            'adamax': Adamax,
            'adadelta': Adadelta,
            'adagrad': Adagrad,
            'rmsprop': RMSprop,
        }[name]
    except:
        raise NotImplementedError(f"Optimizer {name} is not available.")
