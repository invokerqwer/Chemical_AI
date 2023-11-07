import pandas as pd
import torch

dataframe = pd.read_csv('D:\\Nimo77\\Code\\My_CIMG_Bert\\downstream_task\\dataset\\FreeSolv.csv',
                        usecols=['smiles','expt'])
# print(dataframe)
# print(len(dataframe))
# print(dataframe.iloc[0])

# aa, bb = dataframe.iloc[0]
# print(aa)
# print(bb)

dataframe.to_csv('D:\\Nimo77\\Code\\My_CIMG_Bert\\downstream_task\\dataset\\FreeSolv_smile_expt.csv', index=False)
# ————————————————————————————————————————————————————————————————————
# dataaa = pd.read_csv('D:\\Nimo77\\Code\\My_CIMG_Bert\\downstream_task\\dataset\\FreeSolv_smile_expt.csv')
# # print(dataaa)
# # print(len(dataaa))
# # print(dataaa.iloc[0])
# # print(dataaa.iloc[0][1])
# aa, bb = dataaa.iloc[0]
# print(aa)
# print(bb)