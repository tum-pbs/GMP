import torch
import numpy as np
import torch.nn as nn
from timm.layers import trunc_normal_
from einops import rearrange
import torch.distributed.nn as dist_nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import torch.distributed as dist

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}

def matmul_single(fx_mid, slice_weights):
    return fx_mid.T @ slice_weights

def gumbel_softmax(logits, tau=1, hard=False):
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)

    y = logits + gumbel_noise
    y = y / tau
    
    y = F.softmax(y, dim=-1)
    
    if hard:
        _, y_hard = y.max(dim=-1)
        y_one_hot = torch.zeros_like(y).scatter_(-1, y_hard.unsqueeze(-1), 1.0)
        y = (y_one_hot - y).detach() + y
    return y

class Physics_Attention_1D_Eidetic(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU()
        )

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # B N C
        B, N, C = x.shape

        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        
        temperature = self.proj_temperature(x_mid) + self.bias
        temperature = torch.clamp(temperature, min=0.01)
        slice_weights = gumbel_softmax(self.in_project_slice(x_mid), temperature)
        slice_norm = slice_weights.sum(2)  # B H G
        if dist.is_available() and dist.is_initialized():
            dist_nn.all_reduce(slice_norm, op=dist_nn.ReduceOp.SUM)
        slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights).contiguous()
        if dist.is_available() and dist.is_initialized():
            dist_nn.all_reduce(slice_token, op=dist_nn.ReduceOp.SUM)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        out_slice_token = F.scaled_dot_product_attention(q_slice_token, k_slice_token, v_slice_token)

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = int(n_input)
        self.n_hidden = int(n_hidden)
        self.n_output = int(n_output)
        self.n_layers = int(n_layers)
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(self.n_input, self.n_hidden), act())
        self.linear_post = nn.Linear(self.n_hidden, self.n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden), act()) for _ in range(self.n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_plus_block(nn.Module):
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_1D_Eidetic(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                         dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, int(hidden_dim * mlp_ratio), hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        if self.training:
            fx = checkpoint(self.Attn, self.ln_1(fx), use_reentrant=False) + fx
        else:
            fx += self.Attn(self.ln_1(fx))
        if self.training:
            fx = checkpoint(self.mlp, self.ln_2(fx), use_reentrant=False) + fx
        else:
            fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0,
                 n_head=8,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False,
                 condition_size=3
                 ):
        super(Model, self).__init__()
        self.__name__ = 'UniPDE_3D'
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.preprocess = MLP(fun_dim + self.ref * self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0,
                                  res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.embedding = nn.Linear(condition_size, n_hidden)
        self.blocks = nn.ModuleList([Transolver_plus_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      last_layer=(_ == n_layers - 1))
                                     for _ in range(n_layers)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, my_pos):
        # my_pos 1 N 3
        batchsize = my_pos.shape[0]

        gridx = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat([batchsize, 1, self.ref, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 2, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat([batchsize, self.ref, 1, self.ref, 1])
        gridz = torch.tensor(np.linspace(-4, 4, self.ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat([batchsize, self.ref, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).cuda().reshape(batchsize, self.ref ** 3, 3)  # B 4 4 4 3

        pos = torch.sqrt(
            torch.sum((my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2,
                      dim=-1)). \
            reshape(batchsize, my_pos.shape[1], self.ref * self.ref * self.ref).contiguous()
        return pos

    def forward(self, data):
        x, pos, condition = data
        fx, T = None, None
        if self.unified_pos:
            new_pos = self.get_grid(pos)
            x = torch.cat((x, new_pos), dim=-1)

        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        if condition is not None and self.embedding.in_features > 0:
            condition = self.embedding(condition).unsqueeze(1)
            fx = fx + condition
        
        for i, block in enumerate(self.blocks):
            fx = block(fx)
        return fx


if __name__ == "__main__":
    import time
    import torch.nn.functional as F

    B, N, D, in_C, out_C, G = 1, int(32768), 3, 4, 4, 16
    device = "cuda:0"

    # Input tensors matching TransolverPPWrapper forward logic
    x = torch.zeros(B, N, in_C, device=device) # fun_dim is 0
    p = torch.randn(B, N, D, device=device) # space_dim is 3
    x_cat = torch.cat([x, p], dim=-1) # (B, N, D+C) -> (B, N, 3)
    cond = torch.randn(B, G, device=device) # condition_size is 6

    # Instantiate model from transolver_pp_crash.yaml parameters
    model = Model(
        space_dim=D,
        n_layers=6,
        n_hidden=512,
        dropout=0.0,
        n_head=16,
        act="gelu",
        mlp_ratio=1.0,
        fun_dim=in_C,
        out_dim=out_C,
        slice_num=64,
        ref=8,
        unified_pos=False,
        condition_size=G,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params / 1e6:.2f}M")

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)

    def reset_memory_stats():
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)

    print("\nStarting benchmark for 100 random samples (Forward + Backward)...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-8)

    # Warm-up
    print("Warming up for 5 iterations...")
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        out = model((x_cat, p, cond))
        loss = F.mse_loss(out.float(), torch.zeros_like(out).float())
        loss.backward()
        optimizer.step()
    sync()
    print("Warm-up complete.")

    # Count FLOPS
    from torch.utils.flop_counter import FlopCounterMode
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        optimizer.zero_grad(set_to_none=True)
        out = model((x_cat, p, cond))
        loss = F.mse_loss(out.float(), torch.zeros_like(out).float())
        loss.backward()
    flops = flop_counter.get_total_flops()

    times = []
    max_memories = []

    for i in range(10):
        optimizer.zero_grad(set_to_none=True)
        reset_memory_stats()
        sync()

        start_time = time.perf_counter()
        out = model((x_cat, p, cond))
        loss = F.mse_loss(out.float(), torch.zeros_like(out).float())
        loss.backward()
        optimizer.step()
        sync()

        elapsed = time.perf_counter() - start_time
        times.append(elapsed)

        if device.startswith("cuda"):
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            max_memories.append(peak_mem)
        else:
            max_memories.append(0.0)

        if (i + 1) % 10 == 0:
            print(f"Sample {i+1:03d}/100 - Time: {elapsed:.4f} s | Peak Memory: {max_memories[-1]:.2f} GB")

    avg_time = sum(times) / len(times)
    peak_mem_overall = max(max_memories) if max_memories else 0.0

    print("\n--- Benchmark Results (100 Samples) ---")
    print(f"Average samples per second: {1/avg_time:.2f}")
    if device.startswith("cuda"):
        print(f"Max memory usage (peak): {peak_mem_overall:.2f} GB")
    else:
        print("Max memory usage: CUDA not available")
    print(f"FLOPS per sample: {flops / 1e9:.2f} G")