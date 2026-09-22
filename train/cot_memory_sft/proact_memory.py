"""
ProAct 2.0 Memory Module

设计核心:
- Episodic Memory (GRU_long):  全部历史 -> ep_tokens  -> 放入 prompt 前端 (感知)
- Procedural Memory (GRU_short+BN): 近期动作 -> proc_tokens -> 放入 step 之后 (决策)
- Latent State Estimator: epi + proc -> z_t -> str_tokens (放入 prompt 前端)
  训练时有 completion / frontier 监督头; 推理时丢弃
- Graph Encoder (可插拔): 若有 task graph -> graph_tokens
  作为 decision-side 条件 token, 只影响 future / action
  训练时可选 graph dropout 做鲁棒性约束

Modules:
- LatentStateEstimator
- GraphEncoder
- ProActMemory  (统一封装)
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Special token constants
# ============================================================================

EP_SLOT_TOKENS = ["<|ep_0|>", "<|ep_1|>"]
STR_SLOT_TOKENS = ["<|str_0|>", "<|str_1|>"]
PROC_SLOT_TOKENS = ["<|proc_0|>", "<|proc_1|>"]
GRAPH_SLOT_TOKENS = ["<|graph_0|>", "<|graph_1|>", "<|graph_2|>", "<|graph_3|>"]

ALL_MEMORY_TOKENS = EP_SLOT_TOKENS + STR_SLOT_TOKENS + PROC_SLOT_TOKENS + GRAPH_SLOT_TOKENS


def build_ep_stub(tokens: Sequence[str] = EP_SLOT_TOKENS) -> str:
    return "Latent episodic memory:\n" + " ".join(tokens)


def build_str_stub(tokens: Sequence[str] = STR_SLOT_TOKENS) -> str:
    return "Latent structural state:\n" + " ".join(tokens)


def build_proc_stub(tokens: Sequence[str] = PROC_SLOT_TOKENS) -> str:
    return " ".join(tokens)


def build_graph_stub(tokens: Sequence[str] = GRAPH_SLOT_TOKENS) -> str:
    return " ".join(tokens)


# ============================================================================
# LatentStateEstimator
# ============================================================================

class LatentStateEstimator(nn.Module):
    """
    Fuses episodic + procedural GRU states into a task-agnostic
    structural latent z_t, then projects to str_tokens.

    Training-only heads:
      completion_head  -> L_prog  (which steps are done)
      frontier_head    -> L_front (which steps are next)
    """

    def __init__(
        self,
        epi_dim: int = 512,
        proc_dim: int = 64,
        d_struct: int = 256,
        memory_slots: int = 2,
        memory_token_dim: int = 2048,
        max_nodes: int = 2000,
    ) -> None:
        super().__init__()
        self.d_struct = d_struct
        self.memory_slots = memory_slots
        self.memory_token_dim = memory_token_dim
        self.max_nodes = max_nodes

        self.fuse = nn.Sequential(
            nn.Linear(epi_dim + proc_dim, d_struct),
            nn.GELU(),
            nn.LayerNorm(d_struct),
        )
        self.token_proj = nn.Linear(d_struct, memory_slots * memory_token_dim)
        self.completion_head = nn.Linear(d_struct, max_nodes)
        self.frontier_head = nn.Linear(d_struct, max_nodes)

    def forward(
        self, epi_state: torch.Tensor, proc_bn: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.fuse(torch.cat([epi_state, proc_bn], dim=-1))
        tokens = self.token_proj(z).view(-1, self.memory_slots, self.memory_token_dim)
        return z, tokens

    def compute_supervision(
        self,
        z: torch.Tensor,
        completed_ids: List[List[int]],
        future_ids: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = z.size(0)
        device = z.device

        comp_target = torch.zeros(B, self.max_nodes, device=device)
        front_target = torch.zeros(B, self.max_nodes, device=device)
        for b in range(B):
            for cid in completed_ids[b]:
                if 0 <= cid < self.max_nodes:
                    comp_target[b, cid] = 1.0
            for fid in future_ids[b]:
                if 0 <= fid < self.max_nodes:
                    front_target[b, fid] = 1.0

        loss_prog = F.binary_cross_entropy_with_logits(
            self.completion_head(z), comp_target
        )
        loss_front = F.binary_cross_entropy_with_logits(
            self.frontier_head(z), front_target
        )
        return loss_prog, loss_front


# ============================================================================
# GraphEncoder
# ============================================================================

class GraphEncoder(nn.Module):
    """
    Encodes a task DAG into a fixed number of graph_tokens.
    Uses learnable node_embed for known tasks or text_proj for novel tasks.
    """

    def __init__(
        self,
        num_nodes: int = 2000,
        node_dim: int = 256,
        num_graph_tokens: int = 4,
        memory_token_dim: int = 2048,
        nhead: int = 4,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.num_graph_tokens = num_graph_tokens

        self.node_embed = nn.Embedding(num_nodes, node_dim, padding_idx=0)
        self.text_proj = nn.Linear(memory_token_dim, node_dim)

        self.edge_attn = nn.MultiheadAttention(
            node_dim, num_heads=nhead, batch_first=True,
        )
        self.readout_queries = nn.Parameter(
            torch.randn(num_graph_tokens, node_dim) * 0.02
        )
        self.readout_attn = nn.MultiheadAttention(
            node_dim, num_heads=nhead, batch_first=True,
        )
        self.out_proj = nn.Linear(node_dim, memory_token_dim)

    def forward(
        self,
        node_ids: torch.Tensor,
        node_mask: torch.Tensor,
        adjacency_mask: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if text_embeds is not None:
            node_embeds = self.text_proj(text_embeds)
        else:
            node_embeds = self.node_embed(node_ids)

        key_padding_mask = ~node_mask.bool()

        nodes, _ = self.edge_attn(
            node_embeds, node_embeds, node_embeds,
            attn_mask=adjacency_mask,
            key_padding_mask=key_padding_mask,
        )

        B = node_ids.size(0)
        queries = self.readout_queries.unsqueeze(0).expand(B, -1, -1)
        graph_tokens, _ = self.readout_attn(
            queries, nodes, nodes,
            key_padding_mask=key_padding_mask,
        )
        return self.out_proj(graph_tokens)


# ============================================================================
# ProActMemory
# ============================================================================

class ProActMemory(nn.Module):
    """
    ProAct 2.0 unified memory:
      episodic GRU_long   -> ep_tokens   (prompt prefix, perception)
      procedural GRU_short -> proc_tokens (mid-sequence, decision)
      latent state est.    -> str_tokens  (prompt prefix, perception)
      graph encoder        -> graph_tokens (decision-side, pluggable)
    """

    def __init__(
        self,
        *,
        num_actions: int,
        hidden_size: int = 512,
        bottleneck_dim: int = 64,
        short_window: int = 4,
        memory_token_dim: int = 2048,
        ep_slots: int = 2,
        str_slots: int = 2,
        proc_slots: int = 2,
        d_struct: int = 256,
        max_nodes: int = 2000,
        use_graph: bool = False,
        num_graph_tokens: int = 4,
        graph_dropout_rate: float = 0.5,
        pad_action_id: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        self.short_window = short_window
        self.memory_token_dim = memory_token_dim
        self.use_graph = use_graph
        self.num_graph_tokens = num_graph_tokens
        self.graph_dropout_rate = graph_dropout_rate
        self.ep_slots = ep_slots
        self.proc_slots = proc_slots

        self.action_embed = nn.Embedding(
            num_actions, hidden_size, padding_idx=pad_action_id,
        )

        # Episodic (long-term)
        self.epi_gru = nn.GRUCell(hidden_size, hidden_size)
        self.epi_norm = nn.LayerNorm(hidden_size)
        self.ep_token_proj = nn.Linear(hidden_size, ep_slots * memory_token_dim)

        # Procedural (short-term + bottleneck)
        self.proc_gru = nn.GRUCell(hidden_size, hidden_size)
        self.proc_norm = nn.LayerNorm(hidden_size)
        self.proc_bn_proj = nn.Linear(hidden_size, bottleneck_dim)
        self.proc_bn_norm = nn.LayerNorm(bottleneck_dim)
        self.proc_token_proj = nn.Linear(
            bottleneck_dim, proc_slots * memory_token_dim,
        )

        # Latent State Estimator
        self.latent_state = LatentStateEstimator(
            epi_dim=hidden_size,
            proc_dim=bottleneck_dim,
            d_struct=d_struct,
            memory_slots=str_slots,
            memory_token_dim=memory_token_dim,
            max_nodes=max_nodes,
        )

        # Graph Encoder (optional)
        if use_graph:
            self.graph_encoder = GraphEncoder(
                num_nodes=max_nodes,
                node_dim=d_struct,
                num_graph_tokens=num_graph_tokens,
                memory_token_dim=memory_token_dim,
            )
        else:
            self.graph_encoder = None

    # ---- GRU helpers ----

    def _encode(
        self,
        gru: nn.GRUCell,
        norm: nn.LayerNorm,
        action_ids: torch.Tensor,
        action_mask: torch.Tensor,
        init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = action_ids.size(0)
        device = action_ids.device
        state = (
            torch.zeros(B, self.hidden_size, dtype=torch.float32, device=device)
            if init is None else init
        )
        for t in range(action_ids.size(1)):
            valid = action_mask[:, t].bool()
            if not valid.any():
                continue
            emb = self.action_embed(action_ids[:, t])
            new = gru(emb[valid], state[valid])
            state = state.clone()
            state[valid] = new
        return norm(state)

    # ---- Episodic ----

    def encode_epi(self, ids, mask, init=None):
        return self._encode(self.epi_gru, self.epi_norm, ids, mask, init)

    def update_epi(self, state, ids, mask):
        return self.encode_epi(ids, mask, init=state)

    def project_ep_tokens(self, epi_state):
        return self.ep_token_proj(epi_state).view(
            -1, self.ep_slots, self.memory_token_dim
        )

    # ---- Procedural ----

    def encode_proc(self, ids, mask, init=None):
        return self._encode(self.proc_gru, self.proc_norm, ids, mask, init)

    def update_proc(self, state, ids, mask):
        return self.encode_proc(ids, mask, init=state)

    def get_proc_bottleneck(self, proc_state):
        return self.proc_bn_norm(self.proc_bn_proj(proc_state))

    def project_proc_tokens(self, proc_state):
        bn = self.get_proc_bottleneck(proc_state)
        return self.proc_token_proj(bn).view(
            -1, self.proc_slots, self.memory_token_dim
        )

    # ---- Latent structural ----

    def compute_str_tokens(self, epi_state, proc_state):
        bn = self.get_proc_bottleneck(proc_state)
        return self.latent_state(epi_state, bn)

    # ---- Graph ----

    def compute_graph_tokens(self, node_ids, node_mask,
                             adjacency_mask=None, text_embeds=None):
        if self.graph_encoder is None:
            return None
        return self.graph_encoder(node_ids, node_mask, adjacency_mask, text_embeds)

    # ---- Unified interface ----

    def compute_all_tokens(
        self,
        epi_state: torch.Tensor,
        proc_state: torch.Tensor,
        graph_data: Optional[Dict[str, torch.Tensor]] = None,
        training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        ep_tokens = self.project_ep_tokens(epi_state)
        proc_tokens = self.project_proc_tokens(proc_state)
        z, str_tokens = self.compute_str_tokens(epi_state, proc_state)

        graph_tokens = None
        actually_use_graph = False
        if self.use_graph and self.graph_encoder is not None and graph_data is not None:
            if training and random.random() < self.graph_dropout_rate:
                pass
            else:
                graph_tokens = self.graph_encoder(
                    node_ids=graph_data["node_ids"],
                    node_mask=graph_data["node_mask"],
                    adjacency_mask=graph_data.get("adjacency_mask"),
                    text_embeds=graph_data.get("text_embeds"),
                )
                actually_use_graph = True

        return {
            "ep_tokens": ep_tokens,
            "str_tokens": str_tokens,
            "proc_tokens": proc_tokens,
            "z": z,
            "graph_tokens": graph_tokens,
            "use_graph": actually_use_graph,
        }


# ============================================================================
# Graph data utilities
# ============================================================================

def build_task_graph_registry(
    taxonomy: Dict,
    action_name_to_id: Dict[str, int],
) -> Dict[str, Dict]:
    registry: Dict[str, Dict] = {}
    for task_name, nodes_dict in taxonomy.items():
        node_ids = []
        edges = []
        for nid_str, node_info in nodes_dict.items():
            name = node_info.get("name", "")
            aid = action_name_to_id.get(name.strip(), 0)
            node_ids.append(aid)

            parents = node_info.get("parent_id")
            if parents is None:
                continue
            if isinstance(parents, (list, tuple)):
                for p in parents:
                    p_name = nodes_dict.get(str(p), {}).get("name", "")
                    p_aid = action_name_to_id.get(p_name.strip(), 0)
                    edges.append((p_aid, aid))
            else:
                p_name = nodes_dict.get(str(parents), {}).get("name", "")
                p_aid = action_name_to_id.get(p_name.strip(), 0)
                edges.append((p_aid, aid))

        if node_ids:
            registry[task_name] = {"node_ids": node_ids, "adjacency": edges}
    return registry


def prepare_graph_tensors(
    graph_info: Dict,
    max_nodes_per_graph: int = 64,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    raw_ids = graph_info["node_ids"][:max_nodes_per_graph]
    N = len(raw_ids)
    node_ids = torch.zeros(1, max_nodes_per_graph, dtype=torch.long, device=device)
    node_mask = torch.zeros(1, max_nodes_per_graph, dtype=torch.long, device=device)
    node_ids[0, :N] = torch.tensor(raw_ids, dtype=torch.long, device=device)
    node_mask[0, :N] = 1

    id_to_pos = {nid: i for i, nid in enumerate(raw_ids)}
    adj = torch.ones(
        1, max_nodes_per_graph, max_nodes_per_graph, dtype=torch.bool, device=device
    )
    for i in range(N):
        adj[0, i, i] = False
    for parent, child in graph_info.get("adjacency", []):
        pi = id_to_pos.get(parent)
        ci = id_to_pos.get(child)
        if pi is not None and ci is not None:
            adj[0, pi, ci] = False
            adj[0, ci, pi] = False

    return {"node_ids": node_ids, "node_mask": node_mask, "adjacency_mask": adj}


# ============================================================================
# Embedding injection hook (same logic as ablation_memory.py)
# ============================================================================

@contextmanager
def inject_memory_token_embeddings(
    embedding_layer: nn.Module,
    *,
    memory_token_ids: Sequence[int],
    memory_values: torch.Tensor,
):
    token_ids = [int(x) for x in memory_token_ids]

    def _hook(module, args, output):
        if not args:
            return output
        input_ids = args[0]
        if not isinstance(input_ids, torch.Tensor):
            return output
        patched = output.clone()
        for slot_idx, tid in enumerate(token_ids):
            positions = (input_ids == tid).nonzero(as_tuple=False)
            for row, col in positions.tolist():
                patched[row, col] = memory_values[row, slot_idx].to(
                    dtype=patched.dtype
                )
        return patched

    handle = embedding_layer.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


# ============================================================================
# Padding utility
# ============================================================================

def pad_action_sequences(
    sequences: Sequence[Sequence[int]],
    pad_id: int = 0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max((len(s) for s in sequences), default=1)
    ids = torch.full(
        (len(sequences), max_len), pad_id, dtype=torch.long, device=device,
    )
    mask = torch.zeros(
        (len(sequences), max_len), dtype=torch.long, device=device,
    )
    for row, seq in enumerate(sequences):
        sl = [int(x) for x in seq]
        if not sl:
            continue
        ids[row, :len(sl)] = torch.tensor(sl, dtype=torch.long, device=device)
        mask[row, :len(sl)] = 1
    return ids, mask


def truncate_to_recent(actions: List[str], window: int) -> List[str]:
    if not actions:
        return []
    return actions[-window:] if len(actions) > window else actions
