from argparse import ArgumentParser

def parse_args():

    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default='pneu_text', help='decoyMNIST, decoyMNIST_cor, pneu, pneu_text, pneu_text_cor')
    parser.add_argument('--model', type=str, default='EarlyRN18', help='EarlyRN18, CNN')
    parser.add_argument('--frac', type=float, default=None, help="Fraction of dataset to take")    
    parser.add_argument('--load_from', type=str, default=None, help='Model checkpoint to continue training from')
    parser.add_argument('--freeze', action='store_true', default=False, help='Freeze encoder')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")    
    
    return parser.parse_args()