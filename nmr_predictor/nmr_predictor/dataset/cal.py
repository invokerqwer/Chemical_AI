import pandas as pd
from predict import NMRPredict
import time
import torch

nmr_predict = NMRPredict(device='cuda') # 或 device='cuda'
smiles = pd.read_csv('E:\\BingLan_Code\\nmr_predictor\\nmr_predictor\dataset\\pretrain_pre_smi.csv')

NMR = []
count = 0
beg_time = time.time()

for i, smile in enumerate(smiles):
    try:
        nmrs = nmr_predict.predict(smile).squeeze(-1).numpy() # 预测一个分子(加上H原子)中每个原子的NMR
    except Exception:
        continue

    NMR.append([smile, nmrs])
    count = count + 1
    if count % 1000 == 0:
        print_step = int(count / 1000)
        end_time = time.time()
        time_T = end_time - beg_time
        beg_time = end_time
        print(f'已计算{print_step}/200; 计算这1000个smiles用时{time_T}')

torch.save(NMR, 'E:\\BingLan_Code\\nmr_predictor\\nmr_predictor\\dataset\\my_train_pre_smi_and_NMR.pth')
print(count)
print(len(smiles)-count)