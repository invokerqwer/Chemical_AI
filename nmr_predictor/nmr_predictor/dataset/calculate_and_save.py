
import torch
import numpy as np
import pandas as pd
from nmr_predictor.predict import NMRPredict

nmr_predict = NMRPredict(device='cuda') # 或 device='cuda'


# dataframe = pd.read_csv('.\\LogS\\LogS.csv', usecols=['smiles','LogS'], encoding='gbk')
# dataframe.to_csv('.\\LogS\\LogS_smi_label.csv', index=False)
# ——————————————————————————————————————————————————————————————————

'''计算Nmr'''
# data = pd.read_csv('.\\LogS\\LogS_smi_Nmr_label.pth')
# smiles = data['smiles']
# lab = data['LogS']
# # print(lab.iloc[0])
# smi_Nmr_label = []
# for i, smi in enumerate(smiles):
#     nmr = nmr_predict.predict(smi)# 预测一个分子(加上H原子)中每个原子的NMR
#     nmr = nmr.squeeze(-1).numpy()
#     label = lab.iloc[i]
#     smi_Nmr_label.append([smi, nmr, label])
#
# torch.save(smi_Nmr_label, '.\\LogS\\LogS_smi_Nmr_label.pth')

# ————————————————————————————————————————————————————————————————————————————————————————————
'''Nmr四舍五入取整，并挑选出[-800, 1000]'''
# ss = torch.load('.\\bace\\smi_Nmr_class.pth')
# cc = []
# for i, SNE in enumerate(ss):
#     smile, nmr, expt = SNE
#     round_nmr = np.round(nmr).astype(int)
#     arr1 = round_nmr[round_nmr <= -800]
#     arr2 = round_nmr[round_nmr >= 1000]
#     if arr1.size > 0 or arr2.size > 0:
#         continue
#     else:
#         cc.append([smile, round_nmr, expt])
# torch.save(cc, '.\\bace\\smi_Nmr_class_round_select.pth')
