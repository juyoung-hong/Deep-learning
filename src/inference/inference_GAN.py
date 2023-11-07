import sys
sys.path.append('src/models')
from GAN import GAN, Generator, Discriminator

from pathlib import Path
import argparse
import pprint
import time

import torch
from torchvision.utils import save_image

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

    G = Generator(
        z_dim=64,
        channels=1,
        height=28,
        width=28,
        device=device
    )
    
    D =  Discriminator(
        channels=1, 
        height=28, 
        width=28
    )

    model = GAN(Generator=G, Discriminator=D).to(device)
    model.eval()

    checkpoint = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoint['model'])

    with torch.no_grad():
        pred = model(1)

    save_path = Path(args.output) / "gan_predict.png"
    save_image(pred, save_path)


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    inference(args)
    end_time = time.time()
    print(f"{end_time - start_time:.2f} seconds elapsed")