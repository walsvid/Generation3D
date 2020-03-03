from scheduler.base.base_trainer import Trainer

from models.zoo.p2m import P2MModel
from models.losses.p2m_loss import P2MLoss
from utils.mesh import Ellipsoid
# from utils.vis.renderer import MeshRenderer


class P2MTrainer(Trainer):
    def init_auxiliary(self):
        # create renderer
        # self.renderer = MeshRenderer(self.options.dataset.camera_f, self.options.dataset.camera_c,
        #                              self.options.dataset.mesh_pos)
        # create ellipsoid
        self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)

    def init_model(self):
        return P2MModel(self.options.model, self.ellipsoid,
                        self.options.dataset.camera_f, self.options.dataset.camera_c,
                        self.options.dataset.mesh_pos)

    def init_loss_functions(self):
        return P2MLoss(self.options.loss, self.ellipsoid).cuda()
