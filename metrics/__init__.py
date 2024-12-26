from metrics.emd import EMDCalculator


def get_metrics(cfg_metrics):
    metrics = {}

    for cfg_metric in cfg_metrics:
        name = cfg_metric.name

        metrics[name] = get_metric(cfg_metric)

    return metrics


def get_metric(cfg_metric):
    name = cfg_metric.pop('name')

    if name == 'emd':
        metric = EMDCalculator(**cfg_metric)
    else:
        raise NotImplementedError(f"Metric {name} is not implemented.")

    return metric
