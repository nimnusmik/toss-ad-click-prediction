import torch
import torch.nn as nn

class CrossNetwork(nn.Module):
    """
    DCN의 Cross Network 부분
    피처 간의 상호작용을 명시적으로 학습하는 네트워크
    """
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        # 각 레이어마다 weight와 bias 파라미터 생성
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=True) for _ in range(num_layers)
        ])
        
    def forward(self, x0):
        """
        x0: 초기 입력 (batch_size, input_dim)
        각 레이어에서 x_l+1 = x_0 * (W_l * x_l + b_l) + x_l 계산
        """
        x_l = x0  # 현재 레이어의 출력
        
        for layer in self.cross_layers:
            # Cross operation: x_l+1 = x_0 * (W_l * x_l + b_l) + x_l
            xl_w = layer(x_l)  # (batch_size, 1)
            x_l = x0 * xl_w + x_l  # element-wise multiplication + residual connection
            
        return x_l

class DCNModel(nn.Module):
    """
    Deep & Cross Network 모델
    Cross Network와 Deep Network를 병렬로 연결한 후 최종 출력층에서 합침
    """
    def __init__(self, d_features, lstm_hidden=64, cross_layers=3, 
                 deep_hidden=[512, 256, 128], dropout=0.3):
        super().__init__()
        
        # 입력 피처 정규화
        self.bn_input = nn.BatchNorm1d(d_features)
        
        # LSTM for sequence features (기존과 동일)
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, batch_first=True)
        
        # 전체 입력 차원 (tabular features + lstm output)
        total_input_dim = d_features + lstm_hidden
        
        # Cross Network: 피처 간 상호작용 학습
        self.cross_net = CrossNetwork(total_input_dim, cross_layers)
        
        # Deep Network: 비선형 변환 학습
        deep_layers = []
        input_dim = total_input_dim
        
        for hidden_dim in deep_hidden:
            deep_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # 배치 정규화 추가
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
            
        self.deep_net = nn.Sequential(*deep_layers)
        
        # 최종 출력층: Cross Network + Deep Network 결합
        final_input_dim = total_input_dim + input_dim  # cross output + deep output
        self.final_layer = nn.Linear(final_input_dim, 1)
        
    def forward(self, x_feats, x_seq, seq_lengths):
        # 1. Tabular features 정규화
        x_tab = self.bn_input(x_feats)
        
        # 2. Sequence features 처리 (LSTM)
        x_seq = x_seq.unsqueeze(-1)  # (batch_size, seq_len, 1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x_seq, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        x_lstm = h_n[-1]  # 마지막 hidden state (batch_size, lstm_hidden)
        
        # 3. 모든 피처 결합
        x_combined = torch.cat([x_tab, x_lstm], dim=1)
        
        # 4. Cross Network와 Deep Network 병렬 처리
        cross_output = self.cross_net(x_combined)  # 피처 상호작용 학습
        deep_output = self.deep_net(x_combined)    # 비선형 변환 학습
        
        # 5. 두 네트워크 출력 결합
        final_input = torch.cat([cross_output, deep_output], dim=1)
        logits = self.final_layer(final_input)
        
        return logits.squeeze(1)
