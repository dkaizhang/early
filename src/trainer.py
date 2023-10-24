import copy
import numpy as np
import os
import torch

from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer():
    def __init__(self, batch_size=32, epochs=0, num_workers=0, writer=None):
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.writer = writer

    def fit(self, model, train_data, val_data, silent=False, save_every=0):
        train_loader = DataLoader(train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        
        best_val_loss = 100000
        step = 0
        for i in range(self.epochs):
            train_loss = 0
            label_len = len(model.get_labels())
            train_c_m = np.zeros((label_len, label_len), dtype=int)
            train_c_m_early = np.zeros((label_len, label_len), dtype=int)

            val_loss = 0
            val_c_m = np.zeros((label_len, label_len), dtype=int)
            val_c_m_early = np.zeros((label_len, label_len), dtype=int)

            val_f1 = 0
            val_f1_early = 0

            for batch_idx, data in enumerate(tqdm(train_loader)):
                batch_loss_dict, c_m, c_m_early = model.training_step(data)
                train_loss += batch_loss_dict["loss"]
                train_c_m += c_m
                train_c_m_early += c_m_early

                self.writer.add_scalar("Loss - train", batch_loss_dict["loss"], step)
                step += 1

            for batch_idx, data in enumerate(tqdm(val_loader)):    
                batch_loss_dict, c_m, c_m_early, batch_perf_dict = model.validation_step(data)
                val_loss += batch_loss_dict["loss"]
                val_c_m += c_m
                val_c_m_early += c_m_early
                val_f1 += batch_perf_dict["f1"]
                val_f1_early += batch_perf_dict["f1_early"]

                self.writer.add_scalar("Loss - val", batch_loss_dict["loss"], step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_at = os.path.join(self.writer.log_dir, 'best-model.pt')
                # torch.save(model.model.state_dict(), save_at)

                # self.writer.add_scalar("Saving best model", 1, step)

            if save_every > 0 and i > 0 and i % save_every == 0:
                save_at = os.path.join(self.writer.log_dir, f'model-{i}.pt')
                torch.save(model.model.state_dict(), save_at)

            self.writer.add_scalar("Acc - train", train_c_m.diagonal().sum() / train_c_m.sum(), step)
            self.writer.add_scalar("Acc - val", val_c_m.diagonal().sum() / val_c_m.sum(), step)

            self.writer.add_scalar("Acc early - train", train_c_m_early.diagonal().sum() / train_c_m_early.sum(), step)
            self.writer.add_scalar("Acc early - val", val_c_m_early.diagonal().sum() / val_c_m_early.sum(), step)

            self.writer.add_scalar("F1 - val", val_f1 / len(val_loader), step)
            self.writer.add_scalar("F1 early - val", val_f1_early / len(val_loader), step)

            if not silent:
                print(f"Epoch: {i}")
                print(f"Train loss: {train_loss / len(train_loader)} \t \t Val loss: {val_loss / len(val_loader)}")
                print(f"Train acc: {train_c_m.diagonal().sum() / train_c_m.sum()} \t \t Val acc: {val_c_m.diagonal().sum() / val_c_m.sum()}")
                print(f"Train early acc: {train_c_m_early.diagonal().sum() / train_c_m_early.sum()} \t \t Val early acc: {val_c_m_early.diagonal().sum() / val_c_m_early.sum()}")
            
            if torch.isnan(train_loss):
                break

        save_at = os.path.join(self.writer.log_dir, 'last-model.pt')
        torch.save(model.model.state_dict(), save_at)

        loss_dict = defaultdict(lambda: 0)
        loss_dict["train_loss"] = train_loss.item() / len(train_loader)
        loss_dict["val_loss"] = val_loss.item() / len(val_loader)

        return loss_dict

    def test(self, model, data, silent=False):

        dataloader = DataLoader(data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        
        test_loss = 0
        label_len = len(model.get_labels())
        
        confusion_matrix = np.zeros((label_len, label_len), dtype=int)
        confusion_matrix_early = np.zeros((label_len, label_len), dtype=int)
        test_f1 = 0
        test_f1_early = 0

        for batch_idx, data in enumerate(tqdm(dataloader)):

            batch_loss_dict, c_m, c_m_early, batch_perf_dict = model.validation_step(data) # add test loss log
            test_loss += batch_loss_dict["loss"]
            confusion_matrix += c_m
            confusion_matrix_early += c_m_early

            test_f1 += batch_perf_dict["f1"]
            test_f1_early += batch_perf_dict["f1_early"]

        acc = confusion_matrix.diagonal().sum()/confusion_matrix.sum()        
        acc_early = confusion_matrix_early.diagonal().sum()/confusion_matrix_early.sum()        

        test_f1 = test_f1 / len(dataloader)
        test_f1 = test_f1_early / len(dataloader)

        perf_dict = {}
        perf_dict["f1"] = test_f1
        perf_dict["f1_early"] = test_f1_early

        if not silent:
            print(f"Test loss: {test_loss / len(dataloader)} \t \t Test acc: {acc} \t \t Test F1: {test_f1}")
            print(f"Test loss: {test_loss / len(dataloader)} \t \t Test acc: {acc_early} \t \t Test F1: {test_f1_early}")

        return test_loss / len(dataloader), confusion_matrix, confusion_matrix_early, perf_dict

    def uncertain(self, model, data):

        dataloader = DataLoader(data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

        ents = []        
        ents_early = []
        for batch_idx, data in enumerate(tqdm(dataloader)):

            ent, ent_early = model.uncertainty_step(data)
            ents.append(ent.detach().cpu())
            ents_early.append(ent_early.detach().cpu())

        return torch.cat(ents, dim=0), torch.cat(ents_early, dim=0)
