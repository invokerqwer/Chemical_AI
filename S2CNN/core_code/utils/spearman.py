# -*- coding: utf-8 -*-
#把计算出的DFT,ML,NN的光谱结果文件dat文件   比较皮尔森相关系数
import os
import shutil
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import sys
plt.rcParams
plt.switch_backend('agg') #linux中请去掉注释：在导入matplotlib的时候指定不需要GUI的backend

pdb_filename='1ha4'

#作CD图

def normal_data(filename,methods,shift_nm):
      cd_x = []
      cd_y = []
      with open(filename,'r') as input_cd:
            for line in input_cd:
                  if line.startswith('#'):
                        pass
                  else:
                        line2=line[:-1].split('\t')
                        x=line[:-1].split('\t')[:1]
                        y=line[:-1].split('\t')[-1]
                        cd_x.append(x)
                        cd_y.append(y)
      cd_x=np.array(cd_x[8751:20400],dtype=np.float64).reshape(-1,1)
      cd_x=10000000/cd_x+shift_nm#转cm-1为nm
      cd_y=np.array(cd_y[8751:20400],dtype=np.float64).reshape(-1,1)
      cd_normal=cd_y/max(cd_y)#归一化
      cd_normal=np.hstack((cd_x,cd_normal))
      return cd_normal,cd_x


#CD_NN,CD_NN_x=normal_data(filename=pdb_filename,methods="-CD-NN",shift_nm=0)

def experimental_data(filename):
      exp_x = []
      exp_y = []
      num = 0
      with open(filename+"_Final_Processed_Spectrum.txt",'r') as input_exp:
            for line in input_exp:
                  num += 1
                  if num < 25:
                        pass
                  else:
                        line2 = line[:-1].split('\t')
                        x = line2[0]
                        y = line2[1] 
                        exp_x.append(x)
                        exp_y.append(y)
      exp_x = np.array(exp_x,dtype=np.float64).reshape(-1,1)
      exp_y = np.array(exp_y,dtype=np.float64).reshape(-1,1)
      exp_normal=exp_y/max(exp_y)
      exp_normal = np.hstack((exp_x,exp_normal))
      return exp_normal,exp_x

'''CD_exp,CD_exp_x= experimental_data(filename=pdb_filename)
CD_exp_y = np.full([len(CD_exp_x),1],np.nan)
new_exp =np.hstack((CD_exp_x,CD_exp_y))
data = np.vstack((CD_NN,new_exp))
data = DataFrame(data)

#######Spearman系数
#######在光谱的横坐标相同的情况下比较y值的变化。当两个光谱横坐标的数量不一致时采用插值的方法补充。
X=np.vstack((CD_NN_x,CD_exp_x))
ind=np.array(X)
a=ind.reshape(1,len(data))
b=np.ndarray.tolist(a)[0]

data.index=b
new_data = data.interpolate(method='values')
new_data = np.array(new_data)
for i in range(0,len(new_data)):
      if new_data[i][0] == 230:
            flag = i
      if new_data[i][0] == 180:
            flag2 = i
newdata = new_data[flag:flag2+1]
print(newdata)

NN_y = []
for i in newdata:
      NN_y.append(i[1])
NN_y = np.array(NN_y).reshape(-1,1)

EXP_y = []
for i in CD_exp:
      if i[0] < 230.2 and i[0] >= 180:
            EXP_y.append(i[1])
EXP_y = np.array(EXP_y).reshape(-1,1)
total_data = np.hstack((NN_y,EXP_y))

total_data = DataFrame(total_data)
spearman=total_data.corr('spearman')[0][1]
print(spearman)'''

