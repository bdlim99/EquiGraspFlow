from losses.mse_loss import MSELoss


def get_losses(cfg_losses):
    losses = {}

    for cfg_loss in cfg_losses:
        name = cfg_loss.name

        losses[name] = get_loss(cfg_loss)

    return losses


def get_loss(cfg_loss):
    name = cfg_loss.pop('name')

    if name == 'mse':
        loss = MSELoss(**cfg_loss)
    else:
        raise NotImplementedError(f"Loss {name} is not implemented.")

    return loss
