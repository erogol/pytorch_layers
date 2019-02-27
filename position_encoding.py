import torch
from torch import nn


class PositionEncoding(nn.Module):
    """
    https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, in_features, dropout, max_len=2000):
        super(PositionEncoding, self).__init__()
        self.max_len = max_len
        self.in_features = in_features
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        self._setup_encoding()

    def _setup_encoding(self):
        position = torch.arange(0, self.max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.in_features, 2).float() *
                             -(math.log(10000.0) / self.in_features))
        pe = torch.zeros(self.max_len, self.in_features)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward_t(self, x, t):
        if x.shape[1] > self.max_len:
            self.max_len = x.shape[1]
            self._setup_encoding()
        x = x + self.pe[t]
        return self.dropout(x)
        
    def forward(self, x):
        if x.shape[1] > self.max_len:
            self.max_len = x.shape[1]
            self._setup_encoding()
        x = x + self.pe[:x.shape[1]]
        return self.dropout(x)