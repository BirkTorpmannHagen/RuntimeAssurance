from pytorch_lightning import Trainer
from datasets import *


from pytorch_lightning.callbacks import ModelCheckpoint

import warnings
from pytorch_lightning.loggers import TensorBoardLogger

warnings.filterwarnings('ignore')

# torch and lightning imports
from torchvision import transforms
from torch.utils.data import DataLoader
from classifier.resnetclassifier import ResNetClassifier


# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.

def train_classifier(train_set, val_set, load_from_checkpoint=None):
    num_classes =  train_set.num_classes
    model =  ResNetClassifier(num_classes, 101, transfer=False, batch_size=32, lr=1e-4).to("cuda")

    if load_from_checkpoint:
        model = ResNetClassifier.load_from_checkpoint(load_from_checkpoint, num_classes=num_classes, resnet_version=101)
    # model = cifarrr
    tb_logger = TensorBoardLogger(save_dir=f"train_logs/{type(train_set).__name__}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"train_logs/{type(train_set).__name__}/checkpoints",
        save_top_k=3,
        verbose=True,
        monitor="val_acc",
        mode="max"
    )

    # ResNetClassifier.load_from_checkpoint("Imagenette_logs/checkpoints/epoch=82-step=24568.ckpt", resnet_version=101, nj
    trainer = Trainer(max_epochs=200, logger=tb_logger, accelerator="gpu",callbacks=checkpoint_callback)
    trainer.fit(model, train_dataloaders=DataLoader(train_set, shuffle=True, batch_size=16, num_workers=24),
                val_dataloaders=DataLoader(val_set, batch_size=16, shuffle=True, num_workers=24))


if __name__ == '__main__':
    #NICO
    size = 512
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomRotation(90),
                        # transforms.ElasticTransform(),
                        transforms.Resize((size,size)),
                        transforms.ToTensor(), ])
    val_trans = transforms.Compose([
                        transforms.Resize((size,size)),
                        transforms.ToTensor(), ])

    # train_set, val_set = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, val_trans, context="dim", seed=0)
    # train_set, val_set = build_imagenette_dataset("../../Datasets/imagenette2", train_trans=trans, val_trans=val_trans)
    train_set, val_set, test, ood_set = build_officehome_dataset("../../Datasets/OfficeHome", train_transform=trans, val_transform=val_trans )
    # train_classifier(train_set, val_set)

    # train_set, val_set, test_set, ood_val_set, ood_test_set = build_office31_dataset("../../Datasets/office31", train_transform=trans, val_transform=val_trans )
    train_classifier(train_set, val_set)
    #train_set, val_set, test_set, ood_val_set, ood_test_set = build_cct_dataset("../../Datasets/CCT", trans, val_trans)
   # train_classifier(train_set, val_set, load_from_checkpoint="train_logs/CCT/checkpoints/epoch=60-step=50813.ckpt")
    # train_set, val_set, ood_set = build_officehome_dataset("../../Datasets/OfficeHome", train_transform=trans, val_transform=val_trans)
    # train_set, test_set,val_set, ood_set = get_pneumonia_dataset("../../Datasets/Pneumonia", trans, val_trans)

    # CIAR10 and MNIST are already trained :D

