import sys
sys.path.append('src/models')
from ACGAN import ACGAN, Generator, Discriminator

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
    parser.add_argument(
        "-l",
        "--label",
        type=int,
        help="Condition of Generator.",
    )
    args = parser.parse_args()

    pprint.pprint(vars(args))

    return args

def inference(args):
    device = args.device

    G = Generator(
        output_shape=[1, 32, 32],
        z_dim=100,
        n_class=10,
        device=device
    )
    
    D = Discriminator(
        input_shape=[1, 32, 32],
        n_class=10
    )

    model = ACGAN(Generator=G, Discriminator=D).to(device)
    model.eval()

    checkpoint = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoint['model'])

    with torch.no_grad():
        pred = model(args.label)

    save_path = Path(args.output) / "acgan_predict.png"
    save_image(pred, save_path)


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    inference(args)
    end_time = time.time()
    print(f"{end_time - start_time:.2f} seconds elapsed")