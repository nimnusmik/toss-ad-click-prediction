import torch
import torch.nn as nn
from typing import List, Optional


class CrossNetwork(nn.Module):
    """
    DCN의 Cross Network 부분
    피처 간의 상호작용을 명시적으로 학습하는 네트워크
    """

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=True) for _ in range(num_layers)
        ])

    def forward(self, x0):
        """
        x0: 초기 입력 (batch_size, input_dim)
        각 레이어에서 x_l+1 = x_0 * (W_l * x_l + b_l) + x_l 계산
        """
        x_l = x0

        for layer in self.cross_layers:
            xl_w = layer(x_l)
            x_l = x0 * xl_w + x_l

        return x_l


class DCNModel(nn.Module):
    """
    Deep & Cross Network 모델 with categorical embeddings.
    """

    def __init__(
        self,
        num_numeric_features: int,
        categorical_cardinalities: Optional[List[int]] = None,
        embedding_dims: Optional[List[int]] = None,
        lstm_hidden: int = 64,
        cross_layers: int = 3,
        deep_hidden: Optional[List[int]] = None,
        dropout: float = 0.3,
        embedding_dropout: float = 0.05,
    ):
        super().__init__()

        deep_hidden = deep_hidden or [512, 256, 128]
        categorical_cardinalities = categorical_cardinalities or []
        embedding_dims = embedding_dims or []

        if len(categorical_cardinalities) != len(embedding_dims):
            raise ValueError("categorical_cardinalities and embedding_dims must align")

        self.num_numeric_features = num_numeric_features
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(cardinality, dim)
            for cardinality, dim in zip(categorical_cardinalities, embedding_dims)
        ])
        self.embedding_dropout = (
            nn.Dropout(embedding_dropout) if self.embedding_layers else nn.Identity()
        )
        self.tabular_embedding_dim = sum(embedding_dims)

        if self.num_numeric_features > 0:
            self.bn_numeric = nn.BatchNorm1d(self.num_numeric_features)
        else:
            self.bn_numeric = nn.Identity()

        tabular_dim = self.num_numeric_features + self.tabular_embedding_dim
        total_input_dim = tabular_dim + lstm_hidden

        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, batch_first=True)
        self.cross_net = CrossNetwork(total_input_dim, cross_layers)

        deep_layers = []
        input_dim = total_input_dim

        for hidden_dim in deep_hidden:
            deep_layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden_dim

        self.deep_net = nn.Sequential(*deep_layers)
        final_input_dim = total_input_dim + input_dim
        self.final_layer = nn.Linear(final_input_dim, 1)

    def forward(self, x_numeric, x_categorical, x_seq, seq_lengths):
        if self.num_numeric_features > 0:
            x_numeric = self.bn_numeric(x_numeric)
        else:
            x_numeric = x_numeric

        if self.embedding_layers:
            if x_categorical is None:
                raise ValueError("x_categorical must be provided when embeddings are defined")
            embedded = [
                emb(x_categorical[:, idx]) for idx, emb in enumerate(self.embedding_layers)
            ]
            x_cat = torch.cat(embedded, dim=1)
            x_cat = self.embedding_dropout(x_cat)
            if self.num_numeric_features > 0:
                x_tab = torch.cat([x_numeric, x_cat], dim=1)
            else:
                x_tab = x_cat
        else:
            x_tab = x_numeric

        x_seq = x_seq.unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x_seq, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        x_lstm = h_n[-1]

        x_combined = torch.cat([x_tab, x_lstm], dim=1)

        cross_output = self.cross_net(x_combined)
        deep_output = self.deep_net(x_combined)

        final_input = torch.cat([cross_output, deep_output], dim=1)
        logits = self.final_layer(final_input)

        return logits.squeeze(1)
