"""Core layers components for ASDGC model execution."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

class RevIN(nn.Module):
    """Reversible instance normalization for multivariate time-series windows."""

    def __init__(self, num_features, eps=1e-05, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        """Normalize or denormalize x.

Args:
    x: Tensor shaped [batch, variables, time].
    mode: Either 'norm' or 'denorm'."""
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Unsupported RevIN mode: {mode}. Use 'norm' or 'denorm'.")
        return x

    def _get_statistics(self, x):
        self.mean = x.mean(dim=-1, keepdim=True).detach()
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        self.std = torch.sqrt(var + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.std
        if self.affine:
            x = x * self.affine_weight.unsqueeze(0).unsqueeze(-1)
            x = x + self.affine_bias.unsqueeze(0).unsqueeze(-1)
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias.unsqueeze(0).unsqueeze(-1)
            x = x / (self.affine_weight.unsqueeze(0).unsqueeze(-1) + self.eps * self.eps)
        x = x * self.std + self.mean
        return x

class AdaptiveScaleGenerator(nn.Module):
    """Generate temporal scales from the input frequency spectrum."""

    def __init__(self, k=5, freq_threshold=0.1, gate_temperature=0.1):
        super().__init__()
        self.k = k
        self.freq_threshold = nn.Parameter(torch.tensor(freq_threshold))
        self.gate_temperature = gate_temperature

    def forward(self, x, seq_len=None):
        B, _, T = x.shape
        seq_len = seq_len if seq_len is not None else T
        fft_result = torch.fft.fft(x, dim=2)
        magnitudes = torch.abs(fft_result)
        avg_magnitudes = magnitudes.mean(dim=1)
        valid_freqs = avg_magnitudes[:, 1:T // 2]
        
        shrunk_freqs = F.relu(valid_freqs - self.freq_threshold)
        topk_vals, topk_indices = torch.topk(shrunk_freqs, k=self.k, dim=1)
        
        tau_gate = torch.sigmoid(topk_vals / self.gate_temperature).mean(dim=1)  # [B]
        scale_counter = Counter()
        for b in range(B):
            for i in range(self.k):
                if topk_vals[b, i].item() > 0:
                    actual_freq = topk_indices[b, i] + 1
                    scale = int(torch.ceil(T / actual_freq).item())
                    if 1 <= scale <= T:
                        scale_counter[scale] += 1
        sorted_scales = [scale for scale, _ in scale_counter.most_common()]
        common_scales = sorted_scales[:self.k]
        if len(common_scales) < self.k:
            supplement_count = self.k - len(common_scales)
            supplement_scales = [seq_len] * supplement_count
            common_scales += supplement_scales
            common_scales = common_scales[:self.k]
        return common_scales, tau_gate

class TemporalAttentionModule(nn.Module):
    

    def __init__(self, hidden_dim):
        super(TemporalAttentionModule, self).__init__()
        self.Wg = nn.Linear(hidden_dim, hidden_dim)
        self.Ug = nn.Linear(hidden_dim, hidden_dim)
        self.vg = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, T, N, K = x.shape
        x_reshaped = x.reshape(batch_size * N, T, K)
        h_mean = x_reshaped.mean(dim=1, keepdim=True)
        et = self.vg(torch.tanh(self.Wg(x_reshaped) + self.Ug(h_mean)))
        alpha = F.softmax(et, dim=1)
        attn_out = (x_reshaped * alpha).sum(dim=1)
        return attn_out.reshape(batch_size, N, K)

class SpatialAttentionModule(nn.Module):
    

    def __init__(self, hidden_dim):
        super(SpatialAttentionModule, self).__init__()
        self.Wd = nn.Linear(hidden_dim, hidden_dim)
        self.Ud = nn.Linear(hidden_dim, hidden_dim)
        self.vd = nn.Linear(hidden_dim, 1)
        self.Wc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        _, N, _ = x.shape
        h_mean = x.mean(dim=1, keepdim=True)
        ek = self.vd(torch.tanh(self.Wd(x) + self.Ud(h_mean)))
        beta = F.softmax(ek, dim=1)
        context = (x * beta).sum(dim=1, keepdim=True).expand(-1, N, -1)
        fused = torch.tanh(self.Wc(torch.cat([x, context], dim=-1)))
        return fused

class AdaptiveGraphLearner(nn.Module):
    """Learn a sparse, segment-level adaptive graph."""

    def __init__(self, node_num, hidden_dim):
        super(AdaptiveGraphLearner, self).__init__()
        self.topk = min(node_num, 10)
        self.hidden_dim = hidden_dim
        self.time_att = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
        self.mlp_e = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
        self.mlp_m = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
        self.trend_gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, time_scale):
        """Return segment adjacency matrices and their time ranges."""
        batch_size, T, N, K = x.shape
        segments = []
        i = 0
        while i < T:
            end_idx = min(i + time_scale, T)
            segments.append((i, end_idx))
            i = end_idx
        num_segments = len(segments)
        seg_list = [x[:, s:e, :, :] for s, e in segments]
        max_seg_len = max((e - s for s, e in segments))
        seg_batch = torch.zeros(batch_size, num_segments, max_seg_len, N, K, device=x.device)
        for idx, (s, e) in enumerate(segments):
            seg_len = e - s
            seg_batch[:, idx, :seg_len, :, :] = seg_list[idx]
        att_logits = self.time_att(seg_batch)
        att_weights = F.softmax(att_logits, dim=2)
        gamma_p_batch = torch.sum(att_weights * seg_batch, dim=2)
        adj_list = []
        for seg_idx in range(num_segments):
            gamma_p = gamma_p_batch[:, seg_idx, :, :]
            sim = torch.matmul(gamma_p, gamma_p.transpose(1, 2)) / K ** 0.5
            topk_sim, topk_idx = torch.topk(sim, k=self.topk, dim=-1)
            gamma_i = gamma_p.unsqueeze(2)
            gamma_j = gamma_p.unsqueeze(1)
            gamma_j_topk = torch.gather(gamma_j.expand(batch_size, N, N, K), dim=2, index=topk_idx.unsqueeze(-1).expand(-1, -1, -1, K))
            gamma_i_topk = gamma_i.expand(batch_size, N, self.topk, K)
            gamma_ij = torch.cat([gamma_i_topk, gamma_j_topk], dim=-1)
            A_hat_sparse = self.mlp_e(gamma_ij).squeeze(-1)
            M_sparse = self.mlp_m(gamma_ij).squeeze(-1)
            trend_corr = topk_sim
            pos_trend = (trend_corr > 0).float()
            neg_trend = (trend_corr <= 0).float()
            A_pos = A_hat_sparse * pos_trend
            A_neg = A_hat_sparse * neg_trend
            A_hat_sparse = self.trend_gamma * A_pos + (1 - self.trend_gamma) * A_neg
            A_sparse = A_hat_sparse * torch.sigmoid(M_sparse)
            A = torch.zeros(batch_size, N, N, device=x.device)
            A = A.scatter_(dim=2, index=topk_idx, src=A_sparse)
            adj_list.append(A)
        return (adj_list, segments)

class MultiHopGraphConv(nn.Module):
    """Apply residual multi-hop graph propagation."""

    def __init__(self, hidden_dim, propagation_depth=2, residual=True):
        super(MultiHopGraphConv, self).__init__()
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.depth = propagation_depth
        self.residual = residual
        self.W = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()) for _ in range(propagation_depth + 1)])
        self.attention_weights = nn.Parameter(torch.ones(propagation_depth + 1))

    def forward(self, x, adj_list, segments):
        num_segments = len(adj_list)
        if num_segments == 0:
            return x
        conv_out = []
        for (start_idx, end_idx), adj in zip(segments, adj_list):
            seg_len = end_idx - start_idx
            seg_x = x[:, start_idx:end_idx, :, :]
            seg_x_mean = seg_x.mean(dim=1, keepdim=True)
            seg_x_residual = seg_x - seg_x_mean
            gamma_p = seg_x_mean.squeeze(1)
            if self.residual:
                residual = gamma_p
            H = [gamma_p]
            for k in range(1, self.depth + 1):
                H_k = self.beta * gamma_p + (1 - self.beta) * torch.bmm(adj, H[k - 1])
                H.append(H_k)
            attention = torch.softmax(self.attention_weights, dim=0)
            out = torch.stack([self.W[k](H[k]) * attention[k] for k in range(self.depth + 1)], dim=1).sum(dim=1)
            if self.residual:
                out = out + residual
            recovered = out.unsqueeze(1) + seg_x_residual
            conv_out.append(recovered)
        return torch.cat(conv_out, dim=1)

class MultiScaleFeatureFusion(nn.Module):
    """Fuse features from multiple temporal scales."""

    def __init__(self, hidden_dim, num_scales):
        super().__init__()
        self.fusion_net = nn.Sequential(nn.Linear(hidden_dim * num_scales, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim))
        self.scale_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

    def forward(self, scale_features):
        B, C, T, N, K = scale_features.shape
        features_flat = scale_features.permute(0, 3, 2, 1, 4).reshape(B * N * T, C, K)
        attn_output, _ = self.scale_attention(features_flat, features_flat, features_flat)
        fused = attn_output.reshape(B, N, T, C * K)
        fused = self.fusion_net(fused)
        return fused.permute(0, 2, 1, 3)

class GlobalStructureLearner(nn.Module):
    """Learn a batch-specific global adjacency matrix."""

    def __init__(self, node_num):
        super(GlobalStructureLearner, self).__init__()
        self.W_gdy = nn.Linear(node_num, node_num)

    def forward(self, x):
        batch_size, N, _ = x.shape
        x_flat = x.reshape(batch_size, N, -1)
        dist = torch.cdist(x_flat, x_flat, p=2)
        dist_norm = dist / dist.sum(dim=-1, keepdim=True)
        A_gdy = torch.sigmoid(self.W_gdy(dist_norm))
        return A_gdy

class GlobalGraphConvolution(nn.Module):
    """Apply graph convolution with the learned global adjacency."""

    def __init__(self, hidden_dim):
        super(GlobalGraphConvolution, self).__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W0 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, A_gdy):
        batch_size, T, N, K = x.shape
        x_reshaped = x.reshape(batch_size * T, N, K)
        A_gdy_rep = A_gdy.unsqueeze(1).repeat(1, T, 1, 1).reshape(batch_size * T, N, N)
        H = torch.tanh(torch.bmm(A_gdy_rep, self.W1(x_reshaped)) + self.W0(x_reshaped))
        return H.reshape(batch_size, T, N, self.W1.out_features)
