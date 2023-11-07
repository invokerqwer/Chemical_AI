import os
import torch
import torch.nn as nn
import time
import math
import random
import datetime
import numpy as np
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
        self.loss_fn = torch.nn.SmoothL1Loss(reduction='mean')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, eps=args.eps, betas=(args.beta1, args.beta2))
        self.records = {'val_losses': []}
        self.save_file = os.path.join(args.save_path, args.save_name)
        self.val_step = args.val_step

        print(f'train set num:{len(train_dataset)}, valid set num:{len(valid_dataset)}, test set num: {len(test_dataset)}')



# ——————————————————————————————————————————————————————————————————————————————————————————————————

    def train_iterations(self):
        # global val_loss, trn_loss
        self.model.train()
        losses = []
        for tra_step, batch_data in enumerate(self.train_dataloader):

            bde_matrix = batch_data['bde_matrix'].to(self.device)
            masked_mol_ids = batch_data["masked_mol_ids"].to(self.device)
            masked_nmr = batch_data["masked_nmr"].to(self.device)
            nmr_labels = batch_data["nmr_labels"].to(self.device)

            output = self.model(masked_mol_ids, masked_nmr, bde_matrix)
            # loss = self.loss_fn(output.squeeze(-1), nmr_labels)  # 计算损失
            loss = torch.sqrt(self.loss_fn(output.squeeze(-1).float(), nmr_labels.float()))
            # loss = self.loss_fn(output.squeeze(-1) / nmr_labels, nmr_labels / nmr_labels)
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
                    masked_mol_ids = batch_data["masked_mol_ids"].to(self.device)
                    masked_nmr = batch_data["masked_nmr"].to(self.device)
                    nmr_labels = batch_data["nmr_labels"].to(self.device)

                    output = self.model(masked_mol_ids, masked_nmr, bde_matrix)
                    loss = self.loss_fn(output.squeeze(-1), nmr_labels)
                    losses.append(loss.cpu().data)
        if mode == 'valid':
            dataloader = self.valid_dataloader
            losses = []
            with torch.no_grad():
                for i, batch_data in enumerate(dataloader):

                    bde_matrix = batch_data['bde_matrix'].to(self.device)
                    masked_mol_ids = batch_data["masked_mol_ids"].to(self.device)
                    masked_nmr = batch_data["masked_nmr"].to(self.device)
                    nmr_labels = batch_data["nmr_labels"].to(self.device)

                    output = self.model(masked_mol_ids, masked_nmr, bde_matrix)
                    loss = self.loss_fn(output.squeeze(-1), nmr_labels)
                    # loss = self.loss_fn(output.squeeze(-1) / nmr_labels, nmr_labels / nmr_labels)
                    losses.append(loss.cpu().data)

        val_loss = np.array(losses).mean()
        return val_loss

    def train(self, epochs=100):
        for epoch in range(epochs):
            begain_time = time.time()
            # ——————————————————————————————————————————————————
            self.model.train()
            losses = []
            for tra_step, batch_data in enumerate(self.train_dataloader):
                bde_matrix = batch_data['bde_matrix'].to(self.device)
                masked_mol_ids = batch_data["masked_mol_ids"].to(self.device)
                masked_nmr = batch_data["masked_nmr"].to(self.device)
                nmr_labels = batch_data["nmr_labels"].to(self.device)

                output = self.model(masked_mol_ids, masked_nmr, bde_matrix)
                loss = self.loss_fn(output.squeeze(-1), nmr_labels)  # 计算损失
                self.optimizer.zero_grad()  # 梯度清零
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新参数
                losses.append(loss.item())
                # ————————————————————————————————————————————————————
                if tra_step > 0 and tra_step % self.val_step == 0 :
                    lo = losses[tra_step - self.val_step : tra_step]
                    train_loss = np.array(lo).mean()
                    val_loss = self.valid_iterations()  # 定义valid_iterations时，已给定默认值mode='valid'
                    self.records['val_losses'].append(val_loss)
                    save_log = ''
                    if val_loss == np.array(self.records['val_losses']).min():
                        save_log = 'Save best model'
                        self.save_model()

                    print(f'epoch: {epoch}, Iteration_Step: {tra_step} / {len(self.train_dataloader)}, '
                          f'train_loss: {train_loss:.5f}, val_loss: {val_loss:.5f}, {save_log}')

                self.model.train()
            end_time = time.time()
            print(f'elapsed_time: {(end_time - begain_time):.2f}')

    def save_model(self):
        torch.save({'model_state_dict': self.model.state_dict()}, self.save_file)


    def load_model(self, path):
        file = torch.load(path)
        self.model.load_state_dict(file['model_state_dict'])

    def test_model(self):
        self.load_model(self.save_file)
        test_loss = self.valid_iterations(mode='test')
        print('test_loss:',test_loss)
