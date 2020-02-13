from .vgg import VGG16
from .resnet import ResNet18
from .resnet50 import resnet50


def get_backbone(options):
    if options.backbone == "vgg16":
        nn_encoder = VGG16(pretrained=options.backbone_pretrained)
    elif options.backbone == "res18":
        nn_encoder = ResNet18(pretrained=options.backbone_pretrained)
    elif options.backbone == "res50":
        nn_encoder = resnet50()
    else:
        raise NotImplementedError("No implemented backbone called '%s' found" % options.backbone)
    return nn_encoder
