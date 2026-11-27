import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpliceGraphGATLayer(nn.Module):
    """Pure-PyTorch Graph Attention (GATv2) layer over a gene's exon-junction splice graph."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.W_src = nn.Linear(in_dim, out_dim, bias=False)
        self.W_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(out_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h_src = self.W_src(h).unsqueeze(1)
        h_dst = self.W_dst(h).unsqueeze(0)
        e = self.attn(F.leaky_relu(h_src + h_dst, 0.2)).squeeze(-1)
        e = e.masked_fill(adj == 0, float("-inf"))
        alpha = self.dropout(F.softmax(e, dim=-1))
        return F.elu(torch.matmul(alpha, self.W_src(h)))


class HybridSpliceGNN_ODE(nn.Module):
    """
    Hybrid Splice-Graph GNN (GATv2) + 5-State Co-Transcriptional Splicing ODE.
    Performs graph attention across coupled exons in a gene and outputs biophysically
    constrained SF activity and isoform degradation rates (k_deg_inc, k_deg_skip).
    """

    def __init__(self, node_feat_dim: int, n_samples: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.gat1 = SpliceGraphGATLayer(node_feat_dim, hidden_dim)
        self.gat2 = SpliceGraphGATLayer(hidden_dim, hidden_dim)
        self.sf_head = nn.Linear(hidden_dim, n_samples)
        self.deg_head = nn.Linear(hidden_dim, 2)

        self.k_tx = 1.0
        self.k_commit = 2.5
        self.k_splice_0 = 2.0
        self.k_skip = 0.5
        self.K_act = 1.0
        self.n_act = 2.0

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.gat1(x, adj)
        h = self.gat2(h, adj)

        sf = 0.01 + 4.99 * torch.sigmoid(self.sf_head(h))
        deg = 0.05 + 0.95 * torch.sigmoid(self.deg_head(h))
        k_deg_inc, k_deg_skip = deg[:, 0:1], deg[:, 1:2]

        act = (sf / self.K_act) ** self.n_act
        k_sp = self.k_splice_0 * (act / (1.0 + act))
        p2 = self.k_tx / (k_sp + self.k_skip)

        m_inc = (k_sp / k_deg_inc) * p2
        m_skip = (self.k_skip / k_deg_skip) * p2
        psi_pred = m_inc / (m_inc + m_skip + 1e-12)
        return psi_pred, sf, k_deg_inc.squeeze(-1), k_deg_skip.squeeze(-1)


def _parse_coords(event_id: str, fallback_idx: int) -> tuple[int, int]:
    nums = [int(n) for n in re.findall(r"\d{4,}", str(event_id))]
    if len(nums) >= 2:
        return min(nums), max(nums)
    return fallback_idx * 1000, fallback_idx * 1000 + 150


def build_splice_graph(
    psi_matrix: np.ndarray,
    mask_matrix: np.ndarray,
    event_ids: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct node feature matrix X and splice-graph adjacency A across exons."""
    n_events, _ = psi_matrix.shape
    coords = [_parse_coords(event_ids[i] if (event_ids and i < len(event_ids)) else "", i) for i in range(n_events)]
    starts = [c for c, _ in coords]
    order = np.argsort(starts)

    adj = np.eye(n_events, dtype=np.float32)
    for rank in range(len(order) - 1):
        u, v = int(order[rank]), int(order[rank + 1])
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    for i in range(n_events):
        s_i, e_i = coords[i]
        for j in range(i + 1, n_events):
            s_j, e_j = coords[j]
            if max(s_i, s_j) <= min(e_i, e_j):
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    feats = []
    for i in range(n_events):
        s_i, e_i = coords[i]
        valid = psi_matrix[i, mask_matrix[i] > 0]
        exon_len = np.log1p(max(1, e_i - s_i)) / 15.0
        mean_p = float(np.mean(valid)) if len(valid) > 0 else 0.5
        std_p = float(np.std(valid)) if len(valid) > 0 else 0.0
        cov = float(np.mean(mask_matrix[i]))
        feats.append([exon_len, mean_p, std_p, cov])

    return np.array(feats, dtype=np.float32), adj


def fit_events_pytorch(
    psi_matrix: np.ndarray,
    mask_matrix: np.ndarray,
    init_sf: np.ndarray | None = None,
    event_ids: list[str] | None = None,
    epochs: int = 300,
    lr: float = 0.015,
) -> dict[str, np.ndarray | str]:
    """Train HybridSpliceGNN_ODE across a gene's exon splice graph on CPU or CUDA GPU."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, n_samples = psi_matrix.shape
    x_np, adj_np = build_splice_graph(psi_matrix, mask_matrix, event_ids=event_ids)
    _, feat_dim = x_np.shape

    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    adj = torch.tensor(adj_np, dtype=torch.float32, device=device)
    target = torch.tensor(psi_matrix, dtype=torch.float32, device=device)
    mask = torch.tensor(mask_matrix, dtype=torch.float32, device=device)

    model = HybridSpliceGNN_ODE(node_feat_dim=feat_dim, n_samples=n_samples).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        opt.zero_grad()
        psi_pred, _, k_inc, k_skip = model(x, adj)
        mse_loss = (((psi_pred - target) ** 2) * mask).sum() / torch.clamp(mask.sum(), min=1.0)
        reg_loss = 0.01 * ((torch.log(k_inc / 0.2)) ** 2 + (torch.log(k_skip / 0.2)) ** 2).mean()
        loss = mse_loss + reg_loss
        loss.backward()
        opt.step()

    with torch.no_grad():
        psi_hat, sf_hat, k_inc_hat, k_skip_hat = model(x, adj)

    return {
        "psi_pred": psi_hat.cpu().numpy(),
        "sf": sf_hat.cpu().numpy(),
        "k_deg_inc": k_inc_hat.cpu().numpy(),
        "k_deg_skip": k_skip_hat.cpu().numpy(),
        "device": str(device),
    }
