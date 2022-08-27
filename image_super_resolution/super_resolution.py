import torch
import argparse
from PIL import Image
import importlib
import torchvision.transforms as transforms
from collections import OrderedDict
import torchvision
import time
import numpy as np
import cv2
import os

def infer(image, cfg, pil_img=True):
    module = importlib.import_module("model.{}".format(args.model))
    net = module.Net(multi_scale=True,
                     group=args.group)

    if not pil_img:
        h, w, _ = image.shape
        image = Image.fromarray(np.uint8(image))

    state_dict = torch.load(args.ckpt_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

    device = torch.device("cpu")
    net = net.to(device)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    lr = image.convert("RGB")
    lr = transform(lr)

    t1 = time.time()
    lr = lr.unsqueeze(0).to(device)
    sr = net(lr, cfg.scale).detach().squeeze(0)
    lr = lr.squeeze(0)
    t2 = time.time()

    bicubic_img = torchvision.transforms.Resize((lr.shape[1]*cfg.scale,lr.shape[2]*cfg.scale),
                                                 interpolation=torchvision.transforms.InterpolationMode.BICUBIC)(lr)

    return sr.numpy(), bicubic_img.numpy()


if __name__ == "__main__":
    MAX_HEIGHT = 256
    MAX_WIDTH = 256
    parser = argparse.ArgumentParser(description="Super Resolution")
    parser.add_argument('--test_image', type=bool, default=True)
    parser.add_argument('--webcam', type=bool, default=False)
    parser.add_argument('--light_version', type=bool, default=False)

    parser.add_argument("--model", type=str, default="carn")
    print(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--ckpt_path", type=str, default=f"{os.path.dirname(__file__)}/../image_super_resolution/CARN-pytorch/carn/model/")
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--shave", type=int, default=20)
    args = parser.parse_args()

    args.ckpt_path(os.path.join(args.ckpt_path, f"{args.model}.pth"))

    img = Image.open('sample.jpg')

    if args.test_image:
        super_res, bicubic = infer(args, img, pil_img=True)


