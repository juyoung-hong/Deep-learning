import sys
sys.path.append('src/models')
from VariationalAutoEncoder import VariationalAutoEncoder, Encoder, Decoder

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

    encoder = Encoder(channels=1)
    decoder = Decoder(channels=1)
    model = VariationalAutoEncoder(Encoder=encoder, Decoder=decoder).to(device)

    checkpoint = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoint['model'])

    pred = model.predict(device)

    save_path = Path(args.output) / "vae_predict.png"
    save_image(pred, save_path)


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    inference(args)
    end_time = time.time()
    print(f"{end_time - start_time:.2f} seconds elapsed")