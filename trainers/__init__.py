from trainers.grasp_trainer import GraspPoseGeneratorTrainer, PartialGraspPoseGeneratorTrainer


def get_trainer(cfg_trainer, device, dataloaders, model, losses, optimizer, metrics, logger):
    name = cfg_trainer.pop('name')

    if name == 'grasp_full':
        trainer = GraspPoseGeneratorTrainer(cfg_trainer, device, dataloaders, model, losses, optimizer, metrics, logger)
    elif name == 'grasp_partial':
        trainer = PartialGraspPoseGeneratorTrainer(cfg_trainer, device, dataloaders, model, losses, optimizer, metrics, logger)
    else:
        raise NotImplementedError(f"Trainer {name} is not implemented.")

    return trainer
