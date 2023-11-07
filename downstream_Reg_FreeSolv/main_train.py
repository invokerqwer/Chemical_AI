import os
import random
import torch
import argparse
import numpy as np
from model import BERT
from torch.utils.data import Dataset, random_split
from pre_trainer import Trainer
from my_datasets import Masked_AtomNmr_Dataset
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Hyper-parameters for masked Nmr pretrain")
'''Bert中的参数'''
parser.add_argument('--atom_vocab_size', default=18, type=int, help='')
parser.add_argument('--nmr_vocab_size', default=1900, type=int, help='nmr的取值范围中的所有整数')
parser.add_argument("--embed_dim", default=128, type=int, help="")
parser.add_argument("--ffn_dim", default=128, type=int, help="")
parser.add_argument("--num_attention_heads", default=4, type=int, help="number of attention heads in bert model")
parser.add_argument("--num_encoder_layers", default=6, type=int, help="number of Encoder layers in bert model")
parser.add_argument("--dropout", default=0.1, type=float)
'''Masked_AtomNmr_Dataset中的参数'''

parser.add_argument('--FreeSolve_Dataset', default='.\\dataset\\FreeSolv_smi_nmrRouSel_bdeNor_label.pth',
                    type=str, help='smile及水合自由能')
parser.add_argument('--mol_fix_len', default=256, type=int, help='截断长度，超过256的分子截断，不够的补零')
'''Trainer中的参数'''
parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), type=str)
parser.add_argument('--epochs', default=100, type=int, help='training epoches')
parser.add_argument('--batch_size', default=16, type=int, help='Trainer中的Dataloder的参数')
parser.add_argument("--lr", default=1e-04, type=float, help="优化器AdamW中的参数")
parser.add_argument("--beta1", default=0.9, type=float, help="优化器AdamW中的参数")
parser.add_argument("--beta2", default=0.999, type=float, help="优化器AdamW中的参数")
parser.add_argument("--eps", default=1e-08, type=float, help="优化器AdamW中的参数")

parser.add_argument("--save_file", default='.\\save_result\\pretrain_D\\FreeSolv_D_v1.pth',
                    type=str, help="file to save the best downstream model")

parser.add_argument("--load_pretrain_par_file", default='.\\save_result\\pretrain_D\\D_best_model_predict.pth',
                    type=str, help="file to save the pretrain best model")

parser.add_argument("--seed", default=722, type=int, help="随机数种子")
#——————————————————————————————————————————————————————————————————————————————————————————————————————————————
def seed_set(seed=1029): #1029
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_model(savefile,model):
    torch.save({'model_state_dict': model.state_dict()}, savefile)

def loadModel(savefile, model):
    state_dict = torch.load(savefile, map_location=args.device)   # map_location=torch.device('cpu')
    model.load_state_dict(state_dict['model_state_dict'],strict=False)
    return model

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————
args = parser.parse_args()

seed_set(seed = args.seed) # 1030, 137

dataset = Masked_AtomNmr_Dataset(corpus_path=args.FreeSolve_Dataset, mol_fix_len=args.mol_fix_len)
Num = len(dataset)
tra_num, val_num = int(Num*0.8), int(Num*0.1)
test_num = Num - (tra_num + val_num)
train_dataset, valid_dataset, test_dataset = random_split(dataset, [tra_num, val_num, test_num])

config_kwargs = {"atom_vocab_size": args.atom_vocab_size,
                 "nmr_vocab_size": args.nmr_vocab_size,
                 "embed_dim": args.embed_dim,
                 "ffn_dim": args.ffn_dim,
                 "head": args.num_attention_heads,
                 "encoder_layers": args.num_encoder_layers,
                 "dropout": args.dropout}

model = BERT(**config_kwargs)
model = model.to(args.device)
# ————————————————————————————————————————————————————————————————————————————————————————
load_model_par = args.load_pretrain_par_file
state_dict = torch.load(load_model_par, map_location=args.device)
model.load_state_dict(state_dict['model_state_dict'], strict=False)

trainer = Trainer(args, model, args.FreeSolve_Dataset, train_dataset, valid_dataset, test_dataset, num_workers=0)
trainer.train(epochs=args.epochs)
# ————————————————————————————————————————————————————————————————————————————————————————
'''test dataset assessment'''
load_model_par_2 = args.save_file
state_dict_2 = torch.load(load_model_par_2, map_location=args.device)
model.load_state_dict(state_dict_2['model_state_dict'], strict=False)

batch_size = len(test_dataset)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, num_workers=0)

model.eval()
with torch.no_grad():
    for i, batch_data in enumerate(test_dataloader):
        bde_matrix = batch_data['bde_matrix'].to(args.device)
        mol_ids = batch_data["mol_ids"].to(args.device)
        nmr = batch_data["nmr"].to(args.device)
        labels = batch_data["labels"].to(args.device)

        output = model(mol_ids, nmr, bde_matrix)
        R2 = r2_score(labels.cpu(), output.squeeze(-1).cpu())
        print(R2)


plt.scatter(labels.cpu(), output.squeeze(-1).cpu())

plt.title(f'FreeSolv_D_v1, seed_{args.seed}')
plt.xlabel("true")
plt.ylabel("pre")
plt.text(-5, -20, f'R2: {R2:.4f}', fontsize=15)

x = [-25,5]
y = [-25,5]
plt.plot(x,y, c = 'black',linewidth = 3)

plt.show()