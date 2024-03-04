import argparse
from collections import OrderedDict

import torch


def convert(ckpt):
    final_ckpt = OrderedDict()
    new_ckpt = OrderedDict()
    for k, v in list(ckpt.items()):
        new_v = v

        if 'backbone' not in k:
            continue
        else:
            new_k = k.replace('backbone.', '')

        new_ckpt[new_k] = new_v
    final_ckpt['model'] = new_ckpt
    return final_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert mmdet to rsp backbone.')
    parser.add_argument('src', default=None, help='src model path or url')
    parser.add_argument('dst', default=None, help='save path')
    args = parser.parse_args()
    checkpoint = torch.load(args.src, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    weight = convert(state_dict)
    torch.save(weight, args.dst)
    print(f'Done!!, save to {args.dst}')


if __name__ == '__main__':
    main()
