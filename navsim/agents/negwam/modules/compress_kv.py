import math
import torch
import torch.nn.functional as F


def compute_attention_scores(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
) -> torch.Tensor:

    q_len, q_heads, head_dim = query_states.shape
    kv_len, kv_heads, _ = key_states.shape
    query_group_size = q_heads // kv_heads
    device = query_states.device

    if query_group_size == 1:
        # Standard MHA: one query group per KV head
        # q: [q_heads, q_len, head_dim], k: [kv_heads, head_dim, kv_len]
        return torch.bmm(
            query_states.transpose(0, 1),
            key_states.to(device).permute(1, 2, 0),
        ) / math.sqrt(head_dim)

    else:
        # GQA: query_group_size query heads per KV head
        # Aggregate over groups by taking the max across groups (conservative estimate)
        q_reshaped = query_states.view(q_len, kv_heads, query_group_size, head_dim)
        q_reshaped = q_reshaped.permute(1, 2, 0, 3).contiguous()

        k_perm = key_states.to(device).permute(1, 2, 0)

        attn_max = None
        for g in range(query_group_size):
            q_g = q_reshaped[:, g]  # [kv_heads, q_len, head_dim]
            attn_g = torch.bmm(q_g, k_perm) / math.sqrt(head_dim)  # [kv_heads, q_len, kv_len]
            attn_max = attn_g if attn_max is None else torch.max(attn_max, attn_g)

        return attn_max  # [kv_heads, q_len, kv_len]


def compute_redundancy_scores(key_states: torch.Tensor) -> torch.Tensor:

    k = key_states.permute(1, 0, 2)  # [kv_heads, kv_len, head_dim]
    k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)

    sim = torch.bmm(k_norm, k_norm.transpose(-1, -2))

    kv_len = sim.shape[-1]
    diag_mask = torch.eye(kv_len, device=sim.device, dtype=torch.bool).unsqueeze(0)
    sim.masked_fill_(diag_mask, 0.0)

    # Mean similarity that each token *receives* from all other tokens → [kv_heads, kv_len]
    # dim=2 averages over the "from" dimension of the similarity matrix
    redundancy = sim.mean(dim=2)

    # Softmax-normalize over token dimension (per head)
    return redundancy.softmax(dim=-1)


def selectiveKV_score_tokens(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    kernel_size: int = 7,
    mix_lambda: float = 0.07,
    return_components: bool = False,
) -> torch.Tensor:

    kv_len = key_states.shape[0]
    if kv_len == 0:
        empty = torch.zeros(0, device=key_states.device)
        return (empty, empty, empty) if return_components else empty

    # ── Step 1: Relevance  ──────────────────────────────────
    attn_weights = compute_attention_scores(query_states, key_states)

    # Softmax over KV dim, then average over query positions
    relevance = (
        F.softmax(attn_weights, dim=-1)
        .mean(dim=1)
        .to(query_states.dtype)
    )
    del attn_weights
    torch.cuda.empty_cache()

    # Max-pool smoothing along the token dimension
    # Each token's relevance is raised to the local neighbourhood maximum
    relevance = F.max_pool1d(
        relevance,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        stride=1,
    )

    # ── Step 2: Redundancy  ─────────────────────────────────
    redundancy = compute_redundancy_scores(key_states)

    # ── Step 3: Combined score  ──────────────────────────────
    scores = relevance * mix_lambda - redundancy * (1.0 - mix_lambda)

    token_scores = scores.mean(dim=0)

    if return_components:
        rel_mean = relevance.mean(dim=0).detach()
        red_mean = redundancy.mean(dim=0).detach()
        del relevance, redundancy, scores
        torch.cuda.empty_cache()
        return token_scores, rel_mean, red_mean

    del relevance, redundancy, scores
    torch.cuda.empty_cache()

    return token_scores
