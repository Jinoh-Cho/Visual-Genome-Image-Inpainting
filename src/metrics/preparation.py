from os.path import exists, join
import os

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.nn as nn
import numpy as np

from metrics.inception_net import InceptionV3
import metrics.fid as fid
import metrics.ins as ins
import utils.misc as misc


class LoadEvalModel(object):
    def __init__(self, eval_backbone, world_size, distributed_data_parallel, device):
        super(LoadEvalModel, self).__init__()
        self.eval_backbone = eval_backbone
        self.save_output = misc.SaveOutput()

        if self.eval_backbone == "Inception_V3":
            self.model = InceptionV3().to(device)
        elif self.eval_backbone == "SwAV":
            self.model = torch.hub.load('facebookresearch/swav', 'resnet50').to(device)
            hook_handles = []
            for name, layer in self.model.named_children():
                if name == "fc":
                    handle = layer.register_forward_pre_hook(self.save_output)
                    hook_handles.append(handle)
        else:
            raise NotImplementedError

        if world_size > 1 and distributed_data_parallel:
            misc.make_model_require_grad(self.model)
            self.model = DDP(self.model, device_ids=[device], broadcast_buffers=True)
        elif world_size > 1 and distributed_data_parallel is False:
            self.model = DataParallel(self.model, output_device=device)
        else:
            pass

    def eval(self):
        self.model.eval()

    def get_outputs(self, x):
        if self.eval_backbone == "Inception_V3":
            repres, logits = self.model(x)
        else:
            logits = self.model(x)
            repres = self.save_output.outputs[0][0]
            self.save_output.clear()
        return repres, logits


def prepare_moments_calculate_ins(data_loader, eval_model, splits, cfgs, logger, device):
    disable_tqdm = device != 0
    eval_model.eval()
    moment_dir = join(cfgs.RUN.save_dir, "moments")
    if not exists(moment_dir):
        os.makedirs(moment_dir)
    moment_path = join(moment_dir, cfgs.DATA.name + "_" + cfgs.RUN.ref_dataset + "_" + \
                       cfgs.RUN.eval_backbone + "_moments.npz")
    is_file = os.path.isfile(moment_path)
    if is_file:
        mu = np.load(moment_path)["mu"]
        sigma = np.load(moment_path)["sigma"]
    else:
        if device == 0:
            logger.info("Calculate moments of {ref} dataset using {eval_backbone} model.".\
                        format(ref=cfgs.RUN.ref_dataset, eval_backbone=cfgs.RUN.eval_backbone))
        mu, sigma = fid.calculate_moments(data_loader=data_loader,
                                          generator="N/A",
                                          discriminator="N/A",
                                          eval_model=eval_model,
                                          is_generate=False,
                                          num_generate="N/A",
                                          y_sampler="N/A",
                                          batch_size=cfgs.OPTIMIZATION.batch_size,
                                          z_prior="N/A",
                                          truncation_factor="N/A",
                                          z_dim="N/A",
                                          num_classes=cfgs.DATA.num_classes,
                                          LOSS="N/A",
                                          RUN="N/A",
                                          is_stylegan=False,
                                          generator_mapping=None,
                                          generator_synthesis=None,
                                          device=device,
                                          disable_tqdm=disable_tqdm)

        if device == 0:
            logger.info("Save calculated means and covariances to disk.")
        np.savez(moment_path, **{"mu": mu, "sigma": sigma})
    return mu, sigma
