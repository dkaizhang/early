import numpy as np
import os
import torch

from pathlib import Path
from src.args import parse_args
from src.data import load_data, load_labels
from src.trainer import Trainer
from src.wrapper import ModelWrapper

def main(args):

    device = f"cuda:0" if torch.cuda.is_available() else "cpu"

    data = args.data
    seed = 0
    frac = args.frac
    labels = load_labels(data=data)

    model = ModelWrapper(model=args.model,
                        load_from=args.load_from,
                        pretrained=True,
                        freeze=args.freeze,
                        device=device,
                        labels=labels)

    trainer = Trainer(batch_size=args.batch_size,
                num_workers=4)

    test_data = load_data(data=data, split='test', seed=seed, frac=frac)

    outs, outs_early = trainer.predict(model, data=test_data)
    dir_name = os.path.basename(os.path.dirname(args.load_from))
    np.save(f"earlyPred_{args.model}_{args.data}_{dir_name}.npy", outs_early.numpy())
    np.save(f"latePred_{args.model}_{args.data}_{dir_name}.npy", outs.numpy())

if __name__ == '__main__':
    args = parse_args()
    main(args)