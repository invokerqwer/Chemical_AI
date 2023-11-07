import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pandas as pd
import random
from copy import copy
from utils import Smiles_to_adjoin, Smiles_to_bdeMatrix
from utils import Tokenizer
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection

class Masked_AtomNmr_Dataset(Dataset):
    def __init__(self, corpus_path: str, mol_fix_len=256):
        super().__init__()
        self.data = torch.load(corpus_path)
        self.mol_fix_len = mol_fix_len
        self.tokenizer = Tokenizer()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        smi, nmr, bde = self.data[index]
        mol, bde_matrix = Smiles_to_bdeMatrix(smi, bde, explicit_hydrogens=True)
        nmr = nmr + 900
        if len(mol) <= self.mol_fix_len - 1:
            mol_interim = mol
            nmr_interim = nmr
        else:
            mol_interim = mol[:self.mol_fix_len - 1]
            nmr_interim = nmr[:self.mol_fix_len - 1]

        mol_interim = self.tokenizer.add_special_atom(mol_interim)

        nmr_interim = np.insert(nmr_interim, 0, 16)  # 在nmr最前面加上, 即令global的nmr为16
        nmr_labels = copy(nmr_interim)

        masked_mol, masked_nmr = self.apply_AtomNmr_mask(mol_interim, nmr_interim)
        masked_mol_ids = np.array(self.tokenizer.convert_atoms_to_ids(masked_mol), np.int64)

        bde_matrix_G = np.ones([len(mol)+1,len(mol)+1]) * 1
        bde_matrix_G[1:, 1:] = bde_matrix

        if bde_matrix_G.shape[0] > self.mol_fix_len:
            bde_matrix_G = bde_matrix_G[:self.mol_fix_len, :self.mol_fix_len]
        bde_matrix_G = np.array(bde_matrix_G, np.float)

        return  bde_matrix_G, masked_mol_ids, masked_nmr, nmr_labels

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        bde_matrix_G, masked_mol_ids, masked_nmr, nmr_labels = tuple(zip(*batch))

        batch_size = len(bde_matrix_G)
        shape = [batch_size] + [np.max([seq.shape for seq in bde_matrix_G])] * 2
        dtype = bde_matrix_G[0].dtype

        array = np.full(shape, 0, dtype=dtype)
        for arr, matrix in zip(array, bde_matrix_G):
            arrslice = tuple(slice(dim) for dim in matrix.shape)
            arr[arrslice] = matrix

        bde_matrix_G = torch.from_numpy(array)
        masked_mol_ids = torch.from_numpy(self.pad_mol_nmr(masked_mol_ids, 17))
        masked_nmr = torch.from_numpy(self.pad_mol_nmr(masked_nmr, 17))
        nmr_labels = torch.from_numpy(self.pad_mol_nmr(nmr_labels, 17))

        return {'bde_matrix': bde_matrix_G,
                'masked_mol_ids': masked_mol_ids,
                'masked_nmr': masked_nmr,
                'nmr_labels': nmr_labels}

    def apply_AtomNmr_mask(self, mol: List[str], nmr) -> Tuple[List[str], List[int]]:
            masked_mol = copy(mol)
            masked_nmr = copy(nmr)
            # labels = np.zeros([len(masked_mol)], np.int64) - 1  # torch.nn.CrossEntropyLoss(ignore_index=-1)
            for i, atom in enumerate(masked_mol):
                if atom in (self.tokenizer.start_token):  # 跳过<globle>，它不能被mask
                    continue
                prob = random.random()
                if prob < 0.2:
                    # labels[i] = self.tokenizer.convert_atom_to_id(atom)  #[-1,-1,3,-1,-1,12,-1,……]
                    prob /= 0.2
                    if prob < 0.8:  # 80% random change to mask token
                        atom = self.tokenizer.mask_token
                        masked_nmr[i] = 77
                    elif prob < 0.9:  # 10% chance to change to random token
                        atom = self.tokenizer.convert_id_to_atom(random.randint(1, 14))
                        masked_nmr[i] = random.randint(100, 1800)
                    else:  # 10% chance to keep current token
                        pass
                    masked_mol[i] = atom

            return masked_mol, masked_nmr

    def pad_mol_nmr(self, batch_mol, constant_value=17, dtype=None) -> np.ndarray:
            batch_size = len(batch_mol)
            shape = [batch_size] + np.max([mol.shape for mol in batch_mol], 0).tolist()

            if dtype is None:
                dtype = batch_mol[0].dtype

            if isinstance(batch_mol[0], np.ndarray):
                array = np.full(shape, constant_value, dtype=dtype)
            elif isinstance(batch_mol[0], torch.Tensor):
                array = torch.full(shape, constant_value, dtype=dtype)

            for arr, mol in zip(array, batch_mol):
                arrslice = tuple(slice(dim) for dim in mol.shape)
                arr[arrslice] = mol
            return array

