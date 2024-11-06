import torch

from loaders.acronym import AcronymFullPointCloud, AcronymPartialPointCloud


def get_dataloader(split, cfg_dataloader):
    cfg_dataloader.dataset.split = split

    dataset = get_dataset(cfg_dataloader.dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg_dataloader.batch_size,
        shuffle=cfg_dataloader.get('shuffle', True),
        num_workers=cfg_dataloader.get('num_workers', 8),
        collate_fn=collate_fn
    )

    return dataloader


def get_dataset(cfg_dataset):
    name = cfg_dataset.pop('name')

    if name == 'full':
        dataset = AcronymFullPointCloud(**cfg_dataset)
    elif name == 'partial':
        dataset = AcronymPartialPointCloud(**cfg_dataset)
    else:
        raise NotImplementedError(f"Dataset {name} not implemented.")
    
    return dataset


def collate_fn(batch_original):
    batch_collated = {}

    for key in batch_original[0].keys():
        if key == 'Ts_grasp':
            batch_collated[key] = [sample[key] for sample in batch_original]
        else:
            batch_collated[key] = torch.stack([sample[key] for sample in batch_original])

    return batch_collated
