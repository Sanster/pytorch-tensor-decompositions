import argparse

import pytorch_lightning as pl
import tensorly as tl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.utilities import rank_zero_only
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import MNIST

from decompositions import cp_decomposition_conv_layer, tucker_decomposition_conv_layer


class LitMNIST(pl.LightningModule):
    def __init__(self, net: str, learning_rate: float):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = "./"
        self.learning_rate = learning_rate

        self.transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ]
        )

        self.model = models.vgg11(num_classes=10)

    def decompose(self, method):
        self.model = self.decompose_conv(self.model, method)
        for param in self.model.parameters():
            param.requires_grad = True

    def decompose_conv(self, model, method):
        feature_modules = model.features._modules
        N = len(feature_modules.keys())
        for i, key in enumerate(feature_modules.keys()):
            if i >= N - 2:
                break
            if isinstance(feature_modules[key], torch.nn.modules.conv.Conv2d):
                conv_layer = feature_modules[key]
                if method == "cp":
                    rank = max(conv_layer.weight.data.numpy().shape) // 3
                    decomposed = cp_decomposition_conv_layer(conv_layer, rank)
                elif method == "tucker":
                    decomposed = tucker_decomposition_conv_layer(conv_layer)
                feature_modules[key] = decomposed

        return model

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32, num_workers=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default="vgg11")
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument(
        "--decompose", action="store_true", help="do decompose and fine tune"
    )
    parser.add_argument(
        "--decompose_method", default="tucker", choices=["tucker", "cp"]
    )
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--save_dir", default="models")

    return parser.parse_args()


class TFLogger(TensorBoardLogger):
    def __init__(self, save_dir: str):
        super().__init__(save_dir, name="")

    @property
    def log_dir(self) -> str:
        # 简化保存的逻辑，直接用 save_dir
        return self.save_dir

    @property
    def root_dir(self) -> str:
        # 简化保存的逻辑，直接用 save_dir
        return self.save_dir

    @rank_zero_only
    def save(self) -> None:
        # 不保存 hparams.yaml 文件
        super(TensorBoardLogger, self).save()


if __name__ == "__main__":
    args = parse_args()

    model_args = {
        "net": args.net,
        "learning_rate": args.learning_rate,
    }

    if args.ckpt is not None:
        tl.set_backend("pytorch")
        model = LitMNIST.load_from_checkpoint(args.ckpt, **model_args)
        if args.decompose:
            model.decompose(args.decompose_method)
    else:
        model = LitMNIST(**model_args)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.save_dir,
        filename="mnist-{epoch:02d}-{val_loss:.2f}-{val_acc:.3f}",
        save_top_k=1,
        save_last=True,
        mode="min",
    )

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=args.epoch,
        progress_bar_refresh_rate=20,
        benchmark=True,
        callbacks=[checkpoint_callback],
        logger=TFLogger(args.save_dir),
    )
    trainer.fit(model)
    trainer.test()
