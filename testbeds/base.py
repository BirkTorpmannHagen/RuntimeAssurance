import numpy as np
import torch.nn
from segmentation_models_pytorch.utils.metrics import Accuracy

from classifier.cifarresnet import num_classes
from ooddetectors import *
from datasets.synthetic_shifts import *
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
import torch.nn as nn
import torch.nn
from torch.utils.data import ConcatDataset
from datasets.office31 import build_office31_dataset
from datasets.nico import build_nico_dataset
from datasets.polyps import build_polyp_dataset
from datasets.officehome import build_officehome_dataset
from datasets.office31 import build_office31_dataset
from datasets.cct import build_cct_dataset
from torchvision.transforms import transforms
from glow.plmodules import GlowPL
from glow.model import Glow
from classifier.resnetclassifier import ResNetClassifier
import os
from bias_samplers import RandomSampler, ClassOrderSampler, ClusterSampler, SequentialSampler


from torchmetrics import Accuracy

class BaseTestBed:
    """
    Abstract class for testbeds; feel free to override for your own datasets!
    """
    def __init__(self, batch_size, num_workers=5, mode="normal", sampler="RandomSampler"):
        self.mode=mode
        self.num_workers=5
        self.noise_range = np.linspace(0,0.5, 21)[1:]


        self.batch_size = batch_size
        self.sampler = sampler


    def dl(self, dataset):
        if self.sampler=="RandomSampler":
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
        elif self.sampler=="SequentialSampler":
            return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, sampler=SequentialSampler(dataset))
        elif self.sampler=="ClassOrderSampler":
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True, sampler=ClassOrderSampler(dataset, num_classes))
        elif self.sampler=="ClusterSampler":
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True, sampler=ClusterSampler(dataset, self.classifier, self.batch_size))


    def ind_loader(self):
        return  {"ind_train":self.dl(self.ind_train)}

    def ind_val_loader(self):
        return  {"ind_val":self.dl(self.ind_val)}

    def ind_test_loader(self):
        return  {"ind_test":self.dl(self.ind_test)}

    # def get_ood_dict(self):
    #     return {"OoD Val": self.dl(self.ood_val), "Ood Test": self.dl(self.ood_test)}

    def ood_loaders(self):
        if self.mode=="noise":
            print("noise")
            ood_sets = [self.dl(TransformedDataset(self.ind_test, additive_noise, "noise", noise)) for noise in self.noise_range]
            loaders = dict(zip(["noise_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
        elif self.mode=="dropout":
            ood_sets = [self.dl(TransformedDataset(self.ind_test, random_occlusion, "dropout", noise)) for
                        noise in self.noise_range]
            loaders = dict(zip(["dropout_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
        elif self.mode=="saturation":
            ood_sets = [self.dl(TransformedDataset(self.ind_test, desaturate, "saturation", noise)) for
                        noise in self.noise_range]
            loaders = dict(zip(["contrast_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
        elif self.mode=="brightness":
            ood_sets = [self.dl(TransformedDataset(self.ind_test, brightness_shift, "brightness", noise)) for
                        noise in self.noise_range]
            loaders = dict(zip(["brightness_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
        elif self.mode=="hue":
            ood_sets = [self.dl(TransformedDataset(self.ind_test, hue_shift, "hue", noise)) for
                        noise in self.noise_range]
            loaders = dict(zip(["hue_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
        elif self.mode=="fgsm":
            self.num_workers=1
            ood_sets = [self.dl(TransformedDataset(self.ind_test, targeted_fgsm, "fgsm", noise, model=self.classifier)) for
                        noise in self.noise_range]
            loaders = dict(zip(["adv_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
            return loaders
        elif self.mode=="multnoise":
            ood_sets = [self.dl(TransformedDataset(self.ind_test, multiplicative_noise, "multnoise", noise)) for
                        noise in self.noise_range]
            loaders = dict(zip(["multnoise_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
            return loaders
        elif self.mode=="saltpepper":
            ood_sets = [self.dl(TransformedDataset(self.ind_test, salt_and_pepper, "saltpepper", noise)) for
                        noise in self.noise_range]
            loaders = dict(zip(["saltpepper_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
        elif self.mode=="smear":
            ood_sets = [self.dl(TransformedDataset(self.ind_test, smear, "smear", noise)) for
                        noise in self.noise_range]
            loaders = dict(zip(["smear_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
        elif self.mode=="contrast":
            ood_sets = [self.dl(TransformedDataset(self.ind_test, contrast, "contrast", noise)) for noise in self.noise_range]
            loaders = dict(zip(["contrast_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
        elif self.mode=="normal":
            loaders =  self.get_ood_dict()
        else:
            raise NotImplementedError
        return loaders



    def compute_losses(self, loader, reduce=False):
        if reduce:
            losses = torch.zeros((len(loader), 4))  # loss, acc, class, index
        else:
            losses = torch.zeros((len(loader), self.batch_size, 4)) # loss, acc, class, index

        loss_fn = nn.CrossEntropyLoss(reduction="none")  # still computing loss for each sample, just batched

        for i, data in tqdm(enumerate(loader), total=len(loader)):
            with (torch.no_grad()):
                x = data[0].to("cuda")
                y = data[1].to("cuda")
                yhat = self.classifier(x)
                acc = (torch.argmax(yhat, dim=1)==y).float().cpu()
                loss = loss_fn(yhat, y).cpu()
                idx = data[2]

                if reduce:
                    losses[i] = torch.stack([loss.mean(), acc.mean(), idx.float().mean(), y.float().cpu().mean()])
                else:
                    losses[i] = torch.stack([loss, acc, idx, y.cpu()], dim=1)
                # losses[i]=vec
        if reduce:
            return losses.numpy()

        return losses.flatten(0, 1).numpy()

    def get_encodings(self, dataloader):
        features = np.zeros((len(dataloader), self.batch_size, self.classifier.latent_dim))
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Computing Encodings"):
            x = data[0].cuda()
            with torch.no_grad():
                out = self.classifier.get_encoding(x).detach().cpu().numpy()
            features[i]=out
        features = features.reshape((len(dataloader)*self.batch_size, self.classifier.latent_dim))
        return features
