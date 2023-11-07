#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/6/29 10:12
# @Author  : zhangbc0315@outlook.com
# @File    : predict.py
# @Software: PyCharm

import logging

import torch
from torch_geometric.data import Batch

from nmr_predictor.model import NPModel
# from model import NPModel
from nmr_predictor.handler import Handler
# from handler import Handler


logging.basicConfig(level=logging.INFO)


class NMRPredict:

    def __init__(self, device: str = 'cpu'):
        """

        :param device: 'cpu' 或 'cuda'
        """
        # self._model_fp = "./model-e8_7-re0_127-r20_92.pt"
        self._model_fp = "E:\\BingLan_Code\\New\\nmr_predictor\\nmr_predictor\\model-e8_7-re0_127-r20_92.pt"
        self._model = NPModel(9, 5)
        self._model.eval()
        self._model.load_state_dict(torch.load(self._model_fp, map_location=torch.device(device))['model'])

        self._handler = Handler()

    def predict(self, smiles: str):
        data = self._handler.initialization(smiles)
        with torch.no_grad():
            _, pred = self._model(data)
        return pred

    def predict_smiles_list(self, smiles_list: [str]):
        data_list = [self._handler.initialization(smiles) for smiles in smiles_list]
        batch = Batch.from_data_list(data_list)
        with torch.no_grad():
            _, pred = self._model(batch)
        return pred


if __name__ == "__main__":
    r = NMRPredict().predict("c1ccccc1")
    rs = NMRPredict().predict_smiles_list(["c1ccccc1", "CCO"])
    print(r)
    print(rs)
