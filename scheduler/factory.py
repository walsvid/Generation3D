from .p2m.trainer import P2MTrainer


def get_trainer(options, logger, writer):
    if options.model.name == "pixel2mesh":
        trainer = P2MTrainer(options, logger, writer)
    elif options.model.name == "disn":
        raise NotImplementedError("No implemented trainer called '%s' found" % options.model.name)
    else:
        raise NotImplementedError("No implemented trainer called '%s' found" % options.model.name)
    return trainer
