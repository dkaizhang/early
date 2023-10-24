import torch

from src.args import parse_args
from src.data import load_data, load_labels
from src.trainer import Trainer
from src.wrapper import ModelWrapper
from torch.utils.tensorboard import SummaryWriter

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
                        optimizer='SGD',
                        lr=args.lr,
                        labels=labels)
    
    trainer = Trainer(batch_size=args.batch_size,
                    epochs=20,
                    num_workers=4,
                    writer=SummaryWriter())
    
    train_data = load_data(data=data, split='train', seed=seed, frac=frac)
    val_data = load_data(data=data, split='val', seed=seed, frac=frac)

    trainer.fit(model, train_data, val_data, save_every=100)

if __name__ == '__main__':
    args = parse_args()
    main(args)