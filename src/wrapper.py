import torch
import torch.nn.functional as F
import warnings

from src.model import load_model
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import BinaryF1Score, MulticlassF1Score

class ModelWrapper():
    def __init__(self, 
                    model="EarlyRN18",
                    load_from=None, 
                    pretrained=True,
                    freeze=True, 
                    device='cpu',
                    optimizer='SGD',
                    lr=1e-3,
                    labels=[i for i in range(10)]):

        self.model = load_model(model=model, pretrained=pretrained, freeze=freeze, label_len=len(labels))
        self.device = torch.device(device)
        print(f"Initialised on {self.device}")
        self.lr = lr
        self.labels = labels

        if load_from is not None:
            print('loading from state dict...')
            self.model.load_state_dict(torch.load(load_from, map_location='cpu'))

        if optimizer == 'Adam':
            print('Adam')
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer == 'SGD':
            print('SGD')
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        else:
            print("Invalid optimizer name, exiting...")
            exit(1)

    def training_step(self, data):
        self.model.to(self.device)
        self.model.train()

        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()
        y_hat, y_hat_early = self.model(x)

        task_loss = F.cross_entropy(y_hat, y)
        task_loss_early = F.cross_entropy(y_hat_early, y)

        pred = y_hat.argmax(dim=1, keepdim=True)
        c_m = confusion_matrix(y.cpu(), pred.cpu(), labels=self.labels)

        pred_early = y_hat_early.argmax(dim=1, keepdim=True)
        c_m_early = confusion_matrix(y.cpu(), pred_early.cpu(), labels=self.labels)

        loss = 0.5 * task_loss + 0.5 * task_loss_early
        loss.backward()
        self.optimizer.step()

        loss_dict = defaultdict(lambda: 0)
        loss_dict["loss"] = loss

        return loss_dict, c_m, c_m_early

    def validation_step(self, data):
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            x, y = data
            x = x.to(self.device)
            y = y.to(self.device)

            y_hat, y_hat_early = self.model(x)

            if len(self.labels) == 2:
                f1_scorer = BinaryF1Score().to(self.device)
                with warnings.catch_warnings():
                    # warnings.filterwarnings("ignore","No positive samples in targets, true positive value should be meaningless. Returning zero tensor in true positive score")
                    # warnings.filterwarnings("ignore","No negative samples in targets, false positive value should be meaningless. Returning zero tensor in false positive score")    
                    f1 = f1_scorer(y_hat[:,1], y).detach().item()
                    f1_early = f1_scorer(y_hat_early[:,1], y).detach().item()
            else:
                f1_scorer = MulticlassF1Score(num_classes=len(self.labels)).to(self.device)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore","No positive samples in targets, true positive value should be meaningless. Returning zero tensor in true positive score")
                    warnings.filterwarnings("ignore","No negative samples in targets, false positive value should be meaningless. Returning zero tensor in false positive score")
                    f1 = f1_scorer(y_hat, y).detach().item()
                    f1_early = f1_scorer(y_hat_early, y).detach().item()

            loss = 0.5 * F.cross_entropy(y_hat, y) + 0.5 * F.cross_entropy(y_hat_early, y)

            pred = y_hat.argmax(dim=1, keepdim=True)
            c_m = confusion_matrix(y.cpu(), pred.cpu(), labels=self.labels)

            pred_early = y_hat_early.argmax(dim=1, keepdim=True)
            c_m_early = confusion_matrix(y.cpu(), pred_early.cpu(), labels=self.labels)

            loss_dict = defaultdict(lambda: 0)
            loss_dict["loss"] = loss

            perf_dict = {}
            perf_dict["f1"] = f1
            perf_dict["f1_early"] = f1_early

        return loss_dict, c_m, c_m_early, perf_dict

    def uncertainty_step(self, data):
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            x, y = data
            x = x.to(self.device)
            y = y.to(self.device)

            y_hat, y_hat_early = self.model(x)
            
            # print(y_hat[0])

            probs = F.softmax(y_hat, dim=-1)

            # print(probs[0])

            log_probs = probs.log()
            ent = (-probs*log_probs).sum(dim=-1)

            probs_early = F.softmax(y_hat_early, dim=-1)
            log_probs_early = probs_early.log()
            ent_early = (-probs_early*log_probs_early).sum(dim=-1)

            # exit(0)

        return ent, ent_early

    def prediction_step(self, data):
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            x, y = data
            x = x.to(self.device)
            y = y.to(self.device)

            y_hat, y_hat_early = self.model(x)
            
            _, out = F.softmax(y_hat, dim=-1).topk(1, dim=-1)
            _, out_early = F.softmax(y_hat_early, dim=-1).topk(1, dim=-1)

        return out, out_early

    def get_labels(self):
        return self.labels