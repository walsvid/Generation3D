from .p2m.trainer import P2MTrainer
from .disn.trainer import DISNTrainer
from .threedgan.trainer import ThreeDGANTrainer

from .p2m.predictor import P2MPredictor
from .disn.predictor import DISNPredictor


def get_trainer(options, logger, writer):
    if options.model.name == "pixel2mesh":
        trainer = P2MTrainer(options, logger, writer)
    elif options.model.name == "disn":
        trainer = DISNTrainer(options, logger, writer)
    elif options.model.name == "threedgan":
        trainer = ThreeDGANTrainer(options, logger, writer)
    else:
        raise NotImplementedError("No implemented trainer called '%s' found" % options.model.name)
    return trainer


def get_predictor(options, logger, writer):
    if options.model.name == "pixel2mesh":
        predictor = P2MPredictor(options, logger, writer)
    elif options.model.name == "disn":
        predictor = DISNPredictor(options, logger, writer)
    else:
        raise NotImplementedError("No implemented trainer called '%s' found" % options.model.name)
    return predictor
