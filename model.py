import torch, math
from torch import nn, Tensor, matmul
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class TimeEmbedding(nn.Module):

    """ Adapted from: https://github.com/reml-lab/mTAN/blob/main/src/models.py
    """
    def __init__(self, embed_dim, proj_dim, final_dim):

        """
        Parameters:
            d_model: the dimension of the output of sub-layers in the model
            max_len: the maximum length of the input sequences
        """
        super().__init__()
        self.linear = nn.Linear(1,1) # For the first dimension
        self.periodic = nn.Linear(1, embed_dim-1) # For the remaining dimensions
        # self.time_projection = nn.Linear(embed_dim, proj_dim)

        self.time_projection = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            # nn.ReLU(),
            # nn.Linear(proj_dim, final_dim),
            nn.ReLU()
        )

    def forward(self, tt: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, max_len, d_model]
        """

        t2 = torch.sin(self.periodic(tt))
        t1 = self.linear(tt)
        t = torch.cat([t1, t2], -1)
        t = self.time_projection(t)
        return t


class SelfAttention(nn.Module):

  def __init__(self, embed_dim, proj_dim, final_dim):
    super(SelfAttention, self).__init__()

    self.embed_dim = embed_dim # embedding dimension of the input
    self.proj_dim = proj_dim # dimension of q, k v
    self.final_dim = final_dim
    self.time_embedding = TimeEmbedding(embed_dim = embed_dim, proj_dim = proj_dim, final_dim=final_dim)
    
    self.to_key = nn.Sequential(
        nn.Linear(1, embed_dim), # W_k (turns the input features to key vectors)
        nn.ReLU(),
        nn.Linear(embed_dim, proj_dim),
        # nn.ReLU(),
        # nn.Linear(proj_dim, final_dim),
        nn.ReLU()
        ) 
    
    self.to_value = nn.Sequential(
        nn.Linear(1, embed_dim), # W_k (turns the input features to value vectors)
        nn.ReLU(),
        nn.Linear(embed_dim, proj_dim),
        # nn.ReLU(),
        # nn.Linear(proj_dim, final_dim),
        nn.ReLU()
        )
    
   
  def forward(self, x, tt):
    # Forming the Q, K, V using input vectors
    k = self.to_key(x.unsqueeze(-1)).unsqueeze(-1)
    v = self.to_value(x.unsqueeze(-1)).unsqueeze(-1)
    # Apply time embedding
    t = self.time_embedding(tt)
    time_embeddings = torch.tile(t.unsqueeze(0), (x.shape[1], 1, 1)) # d x N x embed_dim
    A = torch.matmul(torch.permute(k, (1,0,3,2)), time_embeddings.unsqueeze(-1)) # d x N x 1 x 1
    A = torch.div(A, math.sqrt(self.proj_dim))
    attention_weights = nn.functional.softmax(A, dim=0) # d x N x 1 x 1
    h = torch.mul(attention_weights, torch.permute(v, (1,0,2,3))).sum(axis=0).squeeze(-1) # N x model_dim
    return h, attention_weights


class AttentionWithTimeEmbedding(nn.Module): 

    def __init__(self, embed_dim, proj_dim, final_dim):
        super(AttentionWithTimeEmbedding, self).__init__()
        self.attention = SelfAttention(embed_dim=embed_dim, proj_dim=proj_dim, final_dim=final_dim)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential( # simple
            nn.Linear(proj_dim*4, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.Linear(16, 2)
        )

    def forward(self, x, time):
        attn_outputs, attention_weights = self.attention(x, time)
        # out = self.classifier(attn_outputs)
        return attn_outputs, attention_weights


    
class LSTMModelWithTimeEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, proj_dim, final_dim):
        super(LSTMModelWithTimeEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.final_dim = final_dim
        self.lstm = nn.LSTM(input_size=proj_dim, hidden_size=final_dim, batch_first=True)
        self.attention_with_temb = AttentionWithTimeEmbedding(embed_dim=embed_dim, proj_dim=proj_dim, final_dim=final_dim)
        self.classifier = nn.Sequential(
        nn.Linear(128,64),
        nn.LeakyReLU(0.2),
        nn.Dropout(p=0.2),
        nn.Linear(64, 16),
        nn.LeakyReLU(0.2),
        nn.Linear(16, 2),
        )
        
    def forward(self, x):
        N = x.data.shape[0] # n_samples
        h_t = torch.zeros(1, N, self.final_dim, dtype = torch.float32).to(device)
        c_t = torch.zeros(1, N, self.final_dim, dtype = torch.float32).to(device)
        num_steps = x.shape[1]

        embeddings, attention_weights = [], []
        for (time_step, input_t) in zip(range(1, num_steps+1), torch.hsplit(x.data, num_steps)):

            input_t = torch.squeeze(input_t, axis=1)
            time_step = torch.tensor(time_step).unsqueeze(-1).float().to(device)

            emb_t, attn_wt_t = self.attention_with_temb(input_t, time_step)
            out_t, (h_t, c_t) = self.lstm(torch.unsqueeze(emb_t, axis=1), (h_t, c_t))
            embeddings.append(out_t)
            attention_weights.append(attn_wt_t.squeeze(dim=-1))
        
        embeddings = torch.cat(embeddings, dim=1)
        attention_weights = torch.cat(attention_weights, dim=-1)
        out = self.classifier(embeddings)
        return out, attention_weights


class CNN1DModelWithTimeEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, proj_dim, final_dim):
        super(CNN1DModelWithTimeEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.final_dim = final_dim
        self.cnn_layer = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1, padding="same")
        self.attention_with_temb = AttentionWithTimeEmbedding(embed_dim=embed_dim, proj_dim=proj_dim, final_dim=final_dim)
        self.classifier = nn.Sequential(
        nn.Linear(128,64),
        nn.LeakyReLU(0.2),
        nn.Dropout(p=0.2),
        nn.Linear(64, 16),
        nn.LeakyReLU(0.2),
        nn.Linear(16, 2),
        )
    def forward(self, x):

        num_steps = x.shape[1]
        embeddings, attention_weights = [], []

        for (time_step, input_t) in zip(range(1, num_steps+1), torch.hsplit(x.data, num_steps)):
            input_t = torch.squeeze(input_t, axis=1)
            time_step = torch.tensor(time_step).unsqueeze(-1).float().to(device)
            emb_t, attn_wt_t = self.attention_with_temb(input_t, time_step)
            embeddings.append(emb_t.unsqueeze(dim=1))
            attention_weights.append(attn_wt_t.squeeze(dim=-1))
        
        embeddings = torch.cat(embeddings, dim=1)
        attention_weights = torch.cat(attention_weights, dim=-1)

        embeddings = embeddings.transpose(1,2)
        out = self.cnn_layer(embeddings)
        out = out.transpose(1,2)
        out = self.classifier(out)
        return out, attention_weights


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]



class TransformerEncoderWithTimeEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, proj_dim, final_dim):
        super(TransformerEncoderWithTimeEmbedding, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.final_dim = final_dim
        self.output_layer = nn.Linear(64, 2)
        self.positional_encoding_layer = PositionalEncoding(64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.2
            ) #sequence first, batch_first=False
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=2)
        self.attention_with_temb = AttentionWithTimeEmbedding(embed_dim=embed_dim, proj_dim=proj_dim, final_dim=final_dim)

    def forward(self, src):
        num_steps = src.shape[1]

        embeddings, attention_weights = [], []

        for (time_step, input_t) in zip(range(1, num_steps+1), torch.hsplit(src.data, num_steps)):

            input_t = torch.squeeze(input_t, axis=1)
            time_step = torch.tensor(time_step).unsqueeze(-1).float().to(device)

            emb_t, attn_wt_t = self.attention_with_temb(input_t, time_step)
            embeddings.append(emb_t.unsqueeze(dim=1))
            attention_weights.append(attn_wt_t.squeeze(dim=-1))
        
        embeddings = torch.cat(embeddings, dim=1)
        attention_weights = torch.cat(attention_weights, dim=-1)

        embeddings = embeddings.transpose(1,0)
        src = self.positional_encoding_layer(embeddings)
        src = src.transpose(1,0)
        src_mask = self._generate_square_subsequent_mask(len(src)).to(device)
        output = self.encoder(src, src_mask)

        return self.output_layer(output), attention_weights

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask