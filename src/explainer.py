import torch

from torch.autograd import grad
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_input_gradients(model, early, input, labels, keep_grad, device='cpu', label_specific=False):
    model.to(device)
    input = input.to(device)
    
    model.eval()

    input.requires_grad_()
    model.zero_grad()
    y, y_early = model(input)

    if early:
        out = y_early
    else:
        out = y

    # First getting LSE'd logits 
    out = out - torch.logsumexp(out, dim=1, keepdim=True)
    if label_specific:
        out = out[torch.arange(len(labels)), labels] # indexing the entry for the true label

    # create_graph allows a higher derivative to be calc'ed
    if keep_grad:
        input_grad = grad(out, input, grad_outputs=torch.ones_like(out), create_graph=True)
    else:
        input_grad = grad(out, input, grad_outputs=torch.ones_like(out), create_graph=False)
    model.train()

    # input_grad is a tuple with the first element being gradient wrt x 
    # https://stackoverflow.com/questions/54166206/grad-outputs-in-torch-autograd-grad-crossentropyloss
    return input_grad[0]

def get_input_gradients_label_specific(model, early, input, labels, keep_grad, device):
    return get_input_gradients(model, early, input, labels, keep_grad, device, label_specific=True)

class Explainer():
    def __init__(self, batch_size=32, num_workers=0, method="input_gradients", pooling=False, device='cpu'):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pooling = pooling
        self.device = device

        if method == 'input_gradients':
            self.method = get_input_gradients
        elif method == 'input_gradients_label_specific':
            self.method = get_input_gradients_label_specific


    def explain(self, model, early, data):

        loader = DataLoader(data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        attributions = []
        for batch_idx, data in enumerate(tqdm(loader)):
            attribution = self.method(model, early, data[0], data[1], keep_grad=False, device=self.device)

            original_size = attribution.shape[-1]

            if self.pooling:
                num_cells = 14
                kernel_size = attribution.shape[-1] // num_cells
                stride = kernel_size
                pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0)
                attribution = pool(attribution.abs())
                attribution = torch.nn.functional.interpolate(attribution, size=original_size)

            attributions.append(attribution.detach().cpu().clone())
        attributions = torch.cat(attributions, dim=0)
        return attributions
    
