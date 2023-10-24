import numpy as np
import os
import torch

from src.args import parse_args
from src.data import load_data, load_labels
from src.explainer import Explainer
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
    
    explainer = Explainer(batch_size=args.batch_size,
                          num_workers=4,
                          method='input_gradients',
                          pooling=True,
                          device=device)
    
    test_data = load_data(data=data, split='test', seed=seed, frac=frac)

    dir_name = os.path.basename(os.path.dirname(args.load_from))

    early_exp = explainer.explain(model.model, early=True, data=test_data)
    np.save(f"earlyExp_{args.model}_{args.data}_{dir_name}.npy", early_exp.numpy())

    late_exp = explainer.explain(model.model, early=False, data=test_data)
    np.save(f"lateExp_{args.model}_{args.data}_{dir_name}.npy", late_exp.numpy())

if __name__ == '__main__':
    args = parse_args()
    main(args)