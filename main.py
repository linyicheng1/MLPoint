from pytorch_lightning.cli import ArgsType, LightningCLI
from model import MInterface
from data import DInterface
import os


def cli_main():
    cli = LightningCLI(MInterface, DInterface)


if __name__ == '__main__':
    cli_main()
