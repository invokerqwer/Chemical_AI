import os
import random
import torch
import argparse
import numpy as np
from model import BERT
from torch.utils.data import Dataset, random_split
from pre_trainer import Trainer
from my_datasets import Masked_AtomNmr_Dataset

def seed_set(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# parser.add_argument('--depth', default=3, type=int)

parser = argparse.ArgumentParser(description="Hyper-parameters for masked Nmr pretrain")
'''Bert中的参数'''
parser.add_argument('--atom_vocab_size', default=18, type=int, help='')
parser.add_argument('--nmr_vocab_size', default=1900, type=int, help='nmr的取值范围中的所有整数')
parser.add_argument("--embed_dim", default=128, type=int, help="")
parser.add_argument("--ffn_dim", default=128, type=int, help='')
parser.add_argument("--num_attention_heads", default=4, type=int, help="number of attention heads in bert model")
parser.add_argument("--num_encoder_layers", default=6, type=int, help="number of Encoder layers in bert model")
parser.add_argument("--dropout", default=0.0, type=float)
'''Masked_AtomNmr_Dataset中的参数'''
parser.add_argument('--AtomNmr_Dataset', default='.\\dataset\\add_smi_nmrRouSel_bdeNor_45.pth',
                    type=str, help='smiles及其nmr,该数据由百成给的nmr_predictor程序算得')
parser.add_argument('--mol_fix_len', default=256, type=int, help='令所有分子的长度都为256，超过的截断，不够的补零')
'''Trainer中的参数'''
parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), type=str)
parser.add_argument('--epochs', default=50, type=int, help='training epoches')
parser.add_argument('--batch_size', default=32, type=int, help='')
parser.add_argument("--lr", default=1e-04, type=float, help="优化器AdamW中的参数")
parser.add_argument("--beta1", default=0.9, type=float, help="优化器AdamW中的参数")
parser.add_argument("--beta2", default=0.999, type=float, help="优化器AdamW中的参数")
parser.add_argument("--eps", default=1e-08, type=float, help="优化器AdamW中的参数")
parser.add_argument("--val_step", default=200, type=int, help="print the train and valid result  every some steps")

parser.add_argument("--save_path", default='.\\save_result', type=str, help="file directory to save the best model")

parser.add_argument("--save_name", default='A5_best_model_08.pth', type=str, help="file name")

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————
args = parser.parse_args()
seed_set(1949)

dataset = Masked_AtomNmr_Dataset(corpus_path=args.AtomNmr_Dataset, mol_fix_len=args.mol_fix_len)
Num = len(dataset)
tra_num, val_num = int(Num*0.95), int(Num*0.05)
test_num = Num - (tra_num + val_num)
train_dataset, valid_dataset, test_dataset = random_split(dataset, [tra_num, val_num, test_num])


config_kwargs = {"atom_vocab_size": args.atom_vocab_size,
                 "nmr_vocab_size": args.nmr_vocab_size,
                 "embed_dim": args.embed_dim,
                 "head": args.num_attention_heads,
                 "encoder_layers": args.num_encoder_layers,
                 "dropout": args.dropout}

model = BERT(**config_kwargs)

trainer = Trainer(args, model, args.AtomNmr_Dataset, train_dataset, valid_dataset, test_dataset, num_workers=0)
trainer.train(epochs=args.epochs)