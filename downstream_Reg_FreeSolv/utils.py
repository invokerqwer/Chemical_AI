from typing import List
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
from collections import OrderedDict

IUPAC_VOCAB = OrderedDict([
    ('<pad>', 17),
    ('H', 1),
    ('C', 2),
    ('N', 3),
    ('O', 4),
    ('F', 5),
    ('S', 6),
    ('Cl', 7),
    ('P', 8),
    ('Br', 9),
    ('B', 10),
    ('I', 11),
    ('Si', 12),
    ('Se', 13),
    ('<unk>', 14),
    ('<mask>', 15),
    ('<global>', 16)])

'''将原子换成对应的数'''
class Tokenizer():

    def __init__(self):

        self.vocab = IUPAC_VOCAB
        self.tokens = list(self.vocab.keys())
        assert self.start_token in self.vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def start_token(self) -> str:
        return "<global>"

    @property
    def mask_token(self) -> str:
        return "<mask>"

    def convert_atom_to_id(self, atom: str) -> int:
        return self.vocab.get(atom, self.vocab['<unk>'])

    def convert_atoms_to_ids(self, mol: List[str]) -> List[int]:
        return [self.convert_atom_to_id(atom) for atom in mol]

    def convert_id_to_atom(self, index: int) -> str:
        return self.tokens[index]

    def convert_ids_to_atoms(self, indices: List[int]) -> List[str]:
        return [self.convert_id_to_atom(idx) for idx in indices]

    def add_special_atom(self, mol: List[str]) -> List[str]:
        return [self.start_token] + mol

#############################################################################

'''由smiles得到邻接矩阵'''
def Smiles_to_adjoin(smiles, explicit_hydrogens=True, canonical_atom_order=False):

    mol = Chem.MolFromSmiles(smiles)

    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)

    num_atoms = mol.GetNumAtoms()

    atom_list = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        atom_list.append(atom.GetSymbol())

    # ——————————————————————————
    adjoin_matrix = np.eye(num_atoms)
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        adjoin_matrix[u, v] = 1.0
        adjoin_matrix[v, u] = 1.0

    return atom_list, adjoin_matrix

def Smiles_to_bdeMatrix(smiles, bond_energies, explicit_hydrogens=True):
    mol = Chem.MolFromSmiles(smiles)

    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)

    num_atoms = mol.GetNumAtoms()

    atom_list = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        atom_list.append(atom.GetSymbol())

    bde_matrix = np.eye(num_atoms) * 1
    for bde, bond in zip(bond_energies, mol.GetBonds()):
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bde_matrix[u, v] = 1.0
        bde_matrix[v, u] = 1.0

    return atom_list, bde_matrix