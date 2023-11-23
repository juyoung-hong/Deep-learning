import sys
sys.path.append('src/models')
from CycleGAN import CycleGAN, Generator, Discriminator
sys.path.append('src/data')
from horse2zebra_datamodule import Horse2ZebraDataModule

from pathlib import Path
import argparse
import pprint
import time

import torch
from torchvision.utils import make_grid, save_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        type=str,
        help="Directory path to save data.",
    )
    parser.add_argument(
        "-d",
        "--device",
        metavar="NAME",
        type=str,
        default='cpu',
        help="Device used inference.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default='4',
        help="Batch size.",
    )
    parser.add_argument(
        "-ckpt",
        "--ckpt_path",
        metavar="DIR",
        type=str,
        help="Directory path has model checkpoint.",
    )
    args = parser.parse_args()

    pprint.pprint(vars(args))

    return args

def inference(args):
    device = args.device

    G_AB = Generator(
        output_shape=[3, 128, 128],
    )
    D_A = Discriminator(
        input_shape=[3, 128, 128],
    )
    modelA = CycleGAN(Generator = G_AB, Discriminator = D_A).to(device)

    G_BA = Generator(
        output_shape=[3, 128, 128],
    )
    D_B = Discriminator(
        input_shape=[3, 128, 128],
    )
    modelB = CycleGAN(Generator = G_BA, Discriminator = D_B).to(device)
    
    modelA.eval()
    modelB.eval()

    checkpoint = torch.load(args.ckpt_path)
    modelA.load_state_dict(checkpoint['modelA'])
    modelB.load_state_dict(checkpoint['modelB'])

    datamodule = Horse2ZebraDataModule(
        data_dir = "data/",
        batch_size = args.batch_size,
        size = [256, 256],
    )
    datamodule.prepare_data()
    datamodule.setup()
    test_dataloader = datamodule.test_dataloader()

    for images in test_dataloader:
        with torch.no_grad():
            real_A = images['A'].to(device)
            real_B = images['B'].to(device)

            fake_A = G_BA(real_B)
            fake_B = G_AB(real_A)
    
    real_A = make_grid(real_A, nrow=args.batch_size, normalize=True, value_range=(-1, 1))
    real_B = make_grid(real_B, nrow=args.batch_size, normalize=True, value_range=(-1, 1))
    fake_A = make_grid(fake_A, nrow=args.batch_size, normalize=True, value_range=(-1, 1))
    fake_B = make_grid(fake_B, nrow=args.batch_size, normalize=True, value_range=(-1, 1))
    grid_image = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_path = Path(args.output) / "cyclegan_predict.png"
    save_image(grid_image, save_path)


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    inference(args)
    end_time = time.time()
    print(f"{end_time - start_time:.2f} seconds elapsed")