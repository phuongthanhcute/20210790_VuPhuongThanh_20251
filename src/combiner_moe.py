import torch
from torch import nn
import torch.nn.functional as F

#  Combiner thiết kế của đồ án
class SparseMoE(nn.Module):
    """
    Cài đặt đơn giản của Mixture of Experts (MoE).
    Gồm n_experts mạng con và một cổng router để chọn top-k chuyên gia.
    """
    def __init__(self, input_dim, output_dim, num_experts=4, k=2):
        super().__init__()
        self.k = k
        self.num_experts = num_experts
        
        # Router: Quyết định expert nào được chọn
        self.router = nn.Linear(input_dim, num_experts)
        
        # Các Experts: Mỗi expert là một mạng feed-forward nhỏ
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        logits = self.router(x)
        
        # Chọn top-k experts có điểm cao nhất
        topk_logits, indices = torch.topk(logits, self.k, dim=-1)
        weights = F.softmax(topk_logits, dim=-1) # (batch, k)
        
        output = torch.zeros_like(x)
        if output.shape[-1] != self.experts[0][0].out_features:
             output = torch.zeros(x.shape[0], self.experts[0][0].out_features, device=x.device)

        # Tính toán output từ các expert được chọn
        for i in range(self.k):
            expert_idx = indices[:, i]
            # Cách này hơi chậm nếu batch lớn, nhưng dễ hiểu. 
            # Thực tế có thể dùng scatter/gather để tối ưu hơn.
            for batch_idx, ex_id in enumerate(expert_idx):
                expert_out = self.experts[ex_id](x[batch_idx].unsqueeze(0))
                output[batch_idx] += weights[batch_idx, i] * expert_out.squeeze(0)
        
        return output

class Combiner(nn.Module):
    """
    Combiner module nâng cấp với Attention và MoE
    """

    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int):
        """
        :param clip_feature_dim: CLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        """
        super(Combiner, self).__init__()
        
        # 1. Input Projections
        self.text_projection_layer = nn.Sequential(
            nn.Linear(clip_feature_dim, projection_dim), # chiếu lên chiều cao hơn (trong đồ án thiết lập là 2560)
            nn.LayerNorm(projection_dim), # Thêm Norm
            nn.ReLU()
        )
        self.image_projection_layer = nn.Sequential(
            nn.Linear(clip_feature_dim, projection_dim), # chiếu lên chiều cao hơn (trong đồ án thiết lập là 2560)
            nn.LayerNorm(projection_dim), # Thêm Norm
            nn.ReLU()
        )

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        # 2. Attention Mechanism (Cross-Attention)
        # Text query Image hoặc ngược lại. Ở đây ta fuse cả 2.
        # Input cho MultiheadAttention cần shape (L, N, E)
        self.attn_layer = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(projection_dim)

        # 3. Mixture of Experts (MoE) thay cho Linear combiner đơn thuần
        # Input là (projection_dim * 2) do concat, output là hidden_dim
        self.moe_layer = SparseMoE(input_dim=projection_dim * 2, output_dim=hidden_dim, num_experts=4, k=2)
        
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)
        self.dropout3 = nn.Dropout(0.5)

        # 4. Element-wise Gating (Thay vì Scalar)
        # Output ra vector cùng chiều với clip_feature_dim để gate từng feature
        self.gate_net = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, clip_feature_dim), # Output vector size = feature dim
            nn.Sigmoid()
        )

        self.logit_scale = 100

    def forward(self, image_features: torch.tensor, text_features: torch.tensor,
                target_features: torch.tensor) -> torch.tensor:
        """
        Giữ nguyên signature như yêu cầu.
        """
        predicted_features = self.combine_features(image_features, text_features)
        target_features = F.normalize(target_features, dim=-1)

        logits = self.logit_scale * predicted_features @ target_features.T
        return logits

    def combine_features(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
        """
        Combine features với luồng xử lý: Project -> Attention -> MoE -> Gated Residual
        """
        # --- 1. Projection ---
        text_projected = self.dropout1(self.text_projection_layer(text_features))
        image_projected = self.dropout2(self.image_projection_layer(image_features))

        # --- 2. Attention Interaction ---
        # Coi Image và Text như một chuỗi có độ dài 2: [Text, Image]
        # Shape: (Batch, 2, Projection_Dim)
        seq = torch.stack([text_projected, image_projected], dim=1)
        
        # Self-Attention giữa Text và Image để chúng "hiểu" nhau hơn
        attn_out, _ = self.attn_layer(seq, seq, seq)
        seq = self.attn_norm(seq + attn_out) # Residual + Norm
        
        # Tách lại ra sau khi đã tương tác qua Attention
        text_attended = seq[:, 0, :]
        image_attended = seq[:, 1, :]

        # --- 3. MoE Fusion ---
        # Nối lại và đưa qua MoE
        raw_combined = torch.cat((text_attended, image_attended), -1)
        combined_features = self.dropout3(F.relu(self.moe_layer(raw_combined)))
        
        # --- 4. Output & Gating ---
        base_output = self.output_layer(combined_features)
        
        # Tính Gate (Element-wise)
        # Gate shape: (Batch, clip_feature_dim) - mỗi chiều feature có trọng số riêng
        gate = self.gate_net(raw_combined) 
        
        # Residual Connection có trọng số (Gated Residual)
        # Công thức: Output + Gate * Text + (1-Gate) * Image
        output = base_output + gate * text_features + (1 - gate) * image_features
        
        return F.normalize(output, dim=-1)