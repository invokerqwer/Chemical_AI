import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

'''embedding层'''

class BERTEmbedding(nn.Module):

    def __init__(self, atom_vocab_size, nmr_vocab_size, embed_dim, dropout=0.1):

        super().__init__()
        self.atom = AtomEmbedding(vocab_size=atom_vocab_size, embed_dim=embed_dim)
        self.nmr = NmrEmbedding(vocab_size=nmr_vocab_size, embed_dim=embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_dim

    def forward(self, mol_ids_list, nmr_list):
        x = self.atom(mol_ids_list) + self.nmr(nmr_list)
        return self.dropout(x)
# ————————————————————————————————
class AtomEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_dim):
        super().__init__(vocab_size, embed_dim, padding_idx=0)
        '''字典中<pad>:0,加上padding_idx=0后，所有分子中padding的原子的embedding都是零向量'''
# ————————————————————————————————
class NmrEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_dim=512):
        super().__init__(vocab_size, embed_dim, padding_idx=0)

#######################################################################################
'''注意力机制'''

class Attention(nn.Module):

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = (scores * mask).float()
            scores = scores.masked_fill(mask == 0, -0.01)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

# ————————————————————————————————————————————————————————————————————————————————
class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embed_dim, dropout = 0.1):
        super().__init__()
        assert embed_dim % head == 0

        self.head = head
        self.h_dim = embed_dim // head

        self.linear_layers = nn.ModuleList([nn.Linear(embed_dim,embed_dim) for _ in range(3)])
        self.output_linear = nn.Linear(embed_dim,embed_dim)

        self.attention = Attention()
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, query, key, value, mask = None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)

        query, key, value = [model(x).view(batch_size, -1, self.head, self.h_dim).transpose(1,2)
                             for model, x in zip(self.linear_layers, (query, key, value))]
        # 将每个头的输出传入到注意力层
        x_value, attn = self.attention(query, key, value, mask = mask, dropout = self.dropout)
        '''每个头的计算结果是4维张量; 将前面1，2两个维度转置回来;
        transpose后面必须使用contiguous方法，不然无法使用view'''
        x_value = x_value.transpose(1,2).contiguous().view(batch_size, -1, self.head * self.h_dim)

        return self.output_linear(x_value)

#######################################################################################

'''前馈全连接层'''
class PositionwiseFeedForward(nn.Module):

    def __init__(self, embed_dim, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(embed_dim, d_ff)
        self.w_2 = nn.Linear(d_ff, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
#--------------------

'''规范化层'''
class LayerNorm(nn.Module):

    def __init__(self,embed_dim, eps = 1e-8):
        super().__init__()

        self.a2 = nn.Parameter(torch.ones(embed_dim))
        self.b2 = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps

    def forward(self,x):

        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)

        return self.a2 * (x - mean) / (std + self.eps) + self.b2
#—————————————————————————————————————————————————————————————————————————————————————

'''子层（残差）连接结构： residual connection followed by a layer norm. Note for code simplicity the norm is first as opposed to last.'''
class SublayerConnection(nn.Module):

    def __init__(self, embed_dim, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

#######################################################################################

'''一个Encoderlayer'''
'''Bidirectional Encoder = Transformer (self-attention); Transformer = MultiHead_Attention + Feed_Forward with sublayer connection'''
class TransformerBlock(nn.Module):

    def __init__(self, head, embed_dim, d_ff, dropout):

        super().__init__()
        self.attention = MultiHeadedAttention(head = head, embed_dim = embed_dim)
        self.input_sublayer = SublayerConnection(embed_dim=embed_dim, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(embed_dim = embed_dim, d_ff = d_ff, dropout = dropout)
        self.output_sublayer = SublayerConnection(embed_dim = embed_dim, dropout = dropout)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda norm_x: self.attention(norm_x, norm_x, norm_x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

#######################################################################################

'''Embedding层 + TransformerBlock = Bert'''
class BERT(nn.Module):

    def __init__(self, atom_vocab_size = 18, nmr_vocab_size = 1000, embed_dim = 128, ffn_dim = 256, head = 4, encoder_layers = 6, dropout=0.1):

        super().__init__()

        self.embedding = BERTEmbedding(atom_vocab_size=atom_vocab_size, nmr_vocab_size=nmr_vocab_size, embed_dim=embed_dim, dropout=dropout)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(head=head, embed_dim=embed_dim, d_ff=embed_dim*4, dropout=dropout) for _ in range(encoder_layers - 1)])

        self.transformer_blocks_downstream = nn.ModuleList([TransformerBlock(head=head, embed_dim=embed_dim, d_ff=embed_dim*4, dropout=dropout) for _ in range(1)])
        self.norm = LayerNorm(embed_dim)

        self.downstream = nn.ModuleList([nn.Linear(embed_dim, ffn_dim), nn.LeakyReLU(), nn.Dropout(p=0.2),
                                         nn.Linear(ffn_dim, ffn_dim), nn.LeakyReLU(), nn.Dropout(p=0.2),
                                         nn.Linear(ffn_dim, 1)])

    def forward(self, mol_ids_list, nmr_list, mask):

        x = self.embedding(mol_ids_list, nmr_list)

        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        for transformer_d in self.transformer_blocks_downstream:
            x = transformer_d(x, mask)

        x = self.norm(x)
        x = x[:, 0, :]


        for ff_down in self.downstream:
            x = ff_down(x)

        return x