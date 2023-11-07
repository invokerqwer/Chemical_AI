import os
import torch
import torch.nn as nn
import time
import math
import random
import datetime
import numpy as np
from sklearn.metrics import r2_score
from my_datasets import Masked_AtomNmr_Dataset
from torch.utils.data.dataloader import DataLoader

class Trainer():
    def __init__(self, args, model, dataset_path, train_dataset, valid_dataset, test_dataset, num_workers=0):
        self.device = args.device
        self.model = model.to(args.device)
        self.dataset = Masked_AtomNmr_Dataset(corpus_path=dataset_path, mol_fix_len=args.mol_fix_len)
        self.train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=self.dataset.collate_fn, shuffle=True, num_workers=num_workers)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=self.dataset.collate_fn, num_workers=num_workers)
        self.test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=self.dataset.collate_fn, num_workers=num_workers)
        # self.loss_fn = torch.nn.SmoothL1Loss(reduction='mean')
        self.loss_fn = torch.nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, eps=args.eps, betas=(args.beta1, args.beta2))
        self.records = {'val_losses': []}
        self.save_file = args.save_file
        print(f'train set num:{len(train_dataset)}, valid set num:{len(valid_dataset)}, test set num: {len(test_dataset)}')

    '''1个epoch中的train'''
    def train_iterations(self):
        self.model.train()
        losses = []
        for i, batch_data in enumerate(self.train_dataloader):

            bde_matrix = batch_data['bde_matrix'].to(self.device)
            mol_ids = batch_data["mol_ids"].to(self.device)
            nmr = batch_data["nmr"].to(self.device)
            labels = batch_data["labels"].to(self.device)

            output = self.model(mol_ids, nmr, bde_matrix)
            # loss = self.loss_fn(output.squeeze(-1) , labels)  # 计算损失
            loss = torch.sqrt(self.loss_fn(output.squeeze(-1).float(), labels.float()))

            self.optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新参数

            losses.append(loss.item())
        trn_loss = np.array(losses).mean()
        return trn_loss

    def valid_iterations(self, mode='valid'):
        self.model.eval()
        if mode == 'test' :
            dataloader = self.test_dataloader
            losses = []
            with torch.no_grad():
                for i, batch_data in enumerate(dataloader):

                    bde_matrix = batch_data['bde_matrix'].to(self.device)
                    mol_ids = batch_data["mol_ids"].to(self.device)
                    nmr = batch_data["nmr"].to(self.device)
                    labels = batch_data["labels"].to(self.device)

                    output = self.model(mol_ids, nmr, bde_matrix)
                    # loss = self.loss_fn(output.squeeze(-1) / labels, labels / labels)
                    # loss = torch.sqrt(self.loss_fn(output.squeeze(-1).float() / labels.float(), labels.float() / labels.float()))
                    loss = torch.sqrt(self.loss_fn(output.squeeze(-1).float(), labels.float()))
                    losses.append(loss.cpu().data)   # loss.cpu()

        if mode == 'valid':
            dataloader = self.valid_dataloader
            losses = []
            with torch.no_grad():
                for i, batch_data in enumerate(dataloader):

                    bde_matrix = batch_data['bde_matrix'].to(self.device)
                    mol_ids = batch_data["mol_ids"].to(self.device)
                    nmr = batch_data["nmr"].to(self.device)
                    labels = batch_data["labels"].to(self.device)

                    output = self.model(mol_ids, nmr, bde_matrix)
                    # loss = self.loss_fn(output.squeeze(-1), labels)
                    loss = torch.sqrt(self.loss_fn(output.squeeze(-1).float(), labels).float())
                    losses.append(loss.cpu().data)

        val_loss = np.array(losses).mean()

        return val_loss

    def train(self, epochs=100):
        for epoch in range(epochs):

            train_loss = self.train_iterations()
            val_loss = self.valid_iterations()
            self.records['val_losses'].append(val_loss)
            print(f'epoch: {epoch}, train_loss: {train_loss:.5f}, val_loss: {val_loss:.5f}')
            save_log = ''
            if val_loss == np.array(self.records['val_losses']).min():
                save_log = 'Save best model'
                self.save_model()
                self.test_model()
                print(f'{save_log}')

    def save_model(self):
        torch.save({'model_state_dict': self.model.state_dict()}, self.save_file)

    def load_model(self, path):
        file = torch.load(path)
        self.model.load_state_dict(file['model_state_dict'])

    def test_model(self):
        self.load_model(self.save_file)
        test_loss = self.valid_iterations(mode='test')
        print('test_loss:',test_loss)



