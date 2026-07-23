"""Core net components for ASDGC model execution."""
import torch
import torch.nn as nn
from layers import AdaptiveScaleGenerator, TemporalAttentionModule, SpatialAttentionModule, AdaptiveGraphLearner, MultiHopGraphConv, MultiScaleFeatureFusion, GlobalStructureLearner, GlobalGraphConvolution, RevIN

class ASDGC(nn.Module):
    """Adaptive Scale Dynamic Graph Convolution network.

Normalization modes:
- global: apply training-set statistics outside the model.
- revin: apply reversible instance normalization inside the model.
- global_revin: combine global normalization with RevIN."""

    def __init__(self, num_variables, seq_len, pred_len, num_scales=5, hidden_dim=32, dropout=0.2, norm_type='global'):
        super(ASDGC, self).__init__()
        self.num_variables = num_variables
        self.seq_len = seq_len
        self.num_scales = num_scales
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        valid_norm_types = ('global', 'revin', 'global_revin')
        if norm_type not in valid_norm_types:
            raise ValueError(f'Unsupported norm_type: {norm_type}. Available values: {valid_norm_types}')
        self.use_revin = norm_type in ('revin', 'global_revin')
        if self.use_revin:
            self.revin = RevIN(num_features=num_variables)
        self.scale_generator = AdaptiveScaleGenerator(k=num_scales)
        
        self.cnn = nn.Conv1d(in_channels=num_variables, out_channels=num_variables * num_scales, kernel_size=1, groups=num_variables)
        
        gru_hidden_half = (hidden_dim + 1) // 2
        self.encoder_gru = nn.GRU(input_size=1, hidden_size=gru_hidden_half, batch_first=True, bidirectional=True)
        self.gru_proj = nn.Linear(gru_hidden_half * 2, hidden_dim)
        self.temporal_attn = TemporalAttentionModule(hidden_dim)
        self.spatial_attn = SpatialAttentionModule(hidden_dim)
        
        self.fuse_z = nn.Linear(hidden_dim * 2, hidden_dim)
        self.scale_graph_learners = nn.ModuleList([AdaptiveGraphLearner(node_num=num_variables, hidden_dim=hidden_dim) for _ in range(num_scales)])
        self.scale_graph_convs = nn.ModuleList([MultiHopGraphConv(hidden_dim=hidden_dim, residual=True) for _ in range(num_scales)])
        self.scale_fusion = MultiScaleFeatureFusion(hidden_dim, num_scales)
        self.global_graph_learner = GlobalStructureLearner(node_num=num_variables)
        self.global_graph_conv = GlobalGraphConvolution(hidden_dim=hidden_dim)
        self.output_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, pred_len))
        self.residual_proj = nn.Sequential(nn.Conv1d(num_scales, hidden_dim, kernel_size=1), nn.GELU())

    def forward(self, x):
        if self.use_revin:
            x = self.revin(x, mode='norm')
        batch_size = x.shape[0]
        time_scales, tau_gate = self.scale_generator(x, seq_len=self.seq_len)
        if len(time_scales) != self.num_scales:
            raise ValueError(f'Generated scale count ({len(time_scales)}) does not match the configured scale count ({self.num_scales}).')
        
        x_cnn = self.cnn(x)  # [B, N * num_scales, T]
        x_cnn = x_cnn.view(batch_size, self.num_variables, self.num_scales, self.seq_len)  # [B, N, k, T]
        x_cnn = x_cnn.permute(0, 2, 3, 1)  # [B, k, T, N]
        
        x_cnn = x_cnn * tau_gate.view(batch_size, 1, 1, 1)
        st_embed_list = []
        for c in range(self.num_scales):
            channel_data = x_cnn[:, c, :, :]  # [B, T, N]
            
            gru_in = channel_data.permute(0, 2, 1).reshape(batch_size * self.num_variables, self.seq_len, 1)
            enc_out, _ = self.encoder_gru(gru_in)  # [B*N, T, 2*gru_hidden_half]
            enc_out = self.gru_proj(enc_out)  # [B*N, T, hidden_dim]
            h_seq = enc_out.reshape(batch_size, self.num_variables, self.seq_len, self.hidden_dim).permute(0, 2, 1, 3)  # [B, T, N, d]
            ta_out = self.temporal_attn(h_seq)  # [B, N, d], Eq. 9-10
            sa_out = self.spatial_attn(ta_out)  # [B, N, d], Eq. 11-12
            sa_broadcast = sa_out.unsqueeze(1).expand(-1, self.seq_len, -1, -1)  # [B, T, N, d]
            scale_feat = torch.tanh(self.fuse_z(torch.cat([h_seq, sa_broadcast], dim=-1)))  # Eq. 13
            st_embed_list.append(scale_feat.unsqueeze(1))
        st_embed = torch.cat(st_embed_list, dim=1)
        st_embed = self.dropout(st_embed)
        scale_outs = []
        adj_list_all = []
        segments_list = []
        for s in range(self.num_scales):
            scale_feat = st_embed[:, s]
            time_scale = time_scales[s]
            adj_list, segments = self.scale_graph_learners[s](scale_feat, time_scale)
            adj_list_all.append(adj_list)
            segments_list.append(segments)
            scale_outs.append(scale_feat)
        enhanced_outs = []
        for s in range(self.num_scales):
            conv_out = self.scale_graph_convs[s](scale_outs[s], adj_list_all[s], segments_list[s])
            enhanced_outs.append(conv_out.unsqueeze(1))
        scale_features = torch.cat(enhanced_outs, dim=1)
        fused_features = self.scale_fusion(scale_features)
        A_gdy = self.global_graph_learner(x)
        global_out = self.global_graph_conv(fused_features, A_gdy)
        residual = self.residual_proj(x_cnn.permute(0, 3, 1, 2).reshape(x_cnn.size(0) * self.num_variables, self.num_scales, self.seq_len)).reshape(x_cnn.size(0), self.num_variables, self.hidden_dim, self.seq_len).permute(0, 3, 1, 2)
        global_out = global_out + residual
        final_feat = global_out[:, -1, :, :]
        pred = self.output_proj(final_feat)
        if self.use_revin:
            pred = self.revin(pred, mode='denorm')
        return pred.permute(0, 2, 1)
