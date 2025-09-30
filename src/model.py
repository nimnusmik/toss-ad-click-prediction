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


class EnhancedCrossNetwork(nn.Module):
    """
    향상된 Cross Network with residual connections and layer normalization
    """

    def __init__(self, input_dim, num_layers, use_layer_norm=True, use_residual=True):
        super().__init__()
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        
        # 각 Cross Layer마다 Linear 변환
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=True) for _ in range(num_layers)
        ])
        
        # Layer Normalization 추가 (선택적)
        if self.use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(input_dim) for _ in range(num_layers)
            ])
        
        # Dropout 추가 (깊은 네트워크에서 정규화)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x0):
        """
        향상된 Cross Network 순전파
        x0: 초기 입력 (batch_size, input_dim)
        """
        x_l = x0

        for i, layer in enumerate(self.cross_layers):
            # 기본 Cross 연산: x_l+1 = x_0 * (W_l * x_l + b_l) + x_l
            xl_w = layer(x_l)  # (batch_size, 1)
            x_cross = x0 * xl_w  # element-wise multiplication with broadcasting
            
            # Residual connection (선택적)
            if self.use_residual:
                x_l = x_cross + x_l
            else:
                x_l = x_cross
            
            # Layer Normalization (선택적)
            if self.use_layer_norm:
                x_l = self.layer_norms[i](x_l)
            
            # Dropout (깊은 네트워크에서 정규화)
            x_l = self.dropout(x_l)

        return x_l


class MultiHeadCrossNetwork(nn.Module):
    """
    Multi-head Cross Network - 여러 개의 Cross path를 병렬로 실행
    """
    
    def __init__(self, input_dim, num_layers, num_heads=3):
        super().__init__()
        self.num_heads = num_heads
        
        # 여러 개의 Cross Network를 병렬로
        self.cross_heads = nn.ModuleList([
            EnhancedCrossNetwork(input_dim, num_layers) 
            for _ in range(num_heads)
        ])
        
        # 결과를 합치는 Linear layer
        self.fusion = nn.Linear(input_dim * num_heads, input_dim)
        
    def forward(self, x0):
        # 각 head에서 나온 결과를 수집
        head_outputs = []
        for head in self.cross_heads:
            head_output = head(x0)
            head_outputs.append(head_output)
        
        # 모든 head 결과를 concat
        combined = torch.cat(head_outputs, dim=1)
        
        # Linear transformation으로 원래 차원으로 변환
        output = self.fusion(combined)
        
        return output


class DCNModelEnhanced(nn.Module):
    """
    향상된 DCN 모델 - 더 깊고 강력한 Cross Network 사용
    """

    def __init__(
        self,
        num_numeric_features: int,
        categorical_cardinalities: Optional[List[int]] = None,
        embedding_dims: Optional[List[int]] = None,
        lstm_hidden: int = 64,
        cross_layers: int = 4,  # 기본값을 4로 증가
        deep_hidden: Optional[List[int]] = None,
        dropout: float = 0.3,
        embedding_dropout: float = 0.05,
        use_enhanced_cross: bool = True,  # 향상된 Cross Network 사용 여부
        use_multi_head: bool = False,     # Multi-head Cross Network 사용 여부
        num_cross_heads: int = 3,         # Multi-head 개수
    ):
        super().__init__()

        deep_hidden = deep_hidden or [512, 256, 128, 64]  # 더 깊게
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

        self.bn_numeric = (
            nn.BatchNorm1d(self.num_numeric_features) 
            if self.num_numeric_features > 0 
            else None
        )

        tabular_dim = self.num_numeric_features + self.tabular_embedding_dim
        total_input_dim = tabular_dim + lstm_hidden

        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, batch_first=True)
        
        # Cross Network 선택
        if use_multi_head:
            self.cross_net = MultiHeadCrossNetwork(
                total_input_dim, cross_layers, num_cross_heads
            )
        elif use_enhanced_cross:
            self.cross_net = EnhancedCrossNetwork(
                total_input_dim, cross_layers, use_layer_norm=True
            )
        else:
            # 기존 CrossNetwork 사용
            #from your_original_module import CrossNetwork  # 원래 모듈에서 import
            self.cross_net = CrossNetwork(total_input_dim, cross_layers)

        # Deep Network - 더 깊게 구성
        deep_layers = []
        input_dim = total_input_dim

        for hidden_dim in deep_hidden:
            deep_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim

        self.deep_net = nn.Sequential(*deep_layers)
        final_input_dim = total_input_dim + input_dim
        self.final_layer = nn.Linear(final_input_dim, 1)

        
    def forward(self, x_numeric, x_categorical, x_seq, seq_lengths):
        # 기존 forward 로직과 동일
        if self.bn_numeric is not None:
            x_numeric = self.bn_numeric(x_numeric)

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

        # 향상된 Cross Network 사용
        cross_output = self.cross_net(x_combined)
        deep_output = self.deep_net(x_combined)

        final_input = torch.cat([cross_output, deep_output], dim=1)
        logits = self.final_layer(final_input)

        return logits.squeeze(1)
