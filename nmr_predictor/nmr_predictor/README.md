# NMR Predictor

预测每个原子的NMR，包括H原子

但是H原子的NMR非常不准，建议不用

输出的非H原子的NMR的index 对应于 原子在smiles中的index

## 依赖

### pytorch
参考官网安装: https://pytorch.org/get-started/locally/

### pytorch geometrics
参考官网安装: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

### rdkit
```bash
conda install -c rdkit rdkit
```

## 运行

```python
from predict import NMRPredict
nmr_predict = NMRPredict(device='cpu') # 或 device='cuda'
nmrs = nmr_predict.predict('c1ccccc1') # 预测一个分子中每个原子的NMR
nmrs_list = nmr_predict.predict_smiles_list(['c1ccccc1', 'CCO']) # 一次预测多个分子的NMR
```
