import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(
            out_features, in_features, device=device, dtype=dtype))

        sigma = math.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(self.W, mean=0, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x @ self.W.transpose(-1, -2)


class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.E = nn.Parameter(torch.empty(
            num_embeddings, embedding_dim, device=device, dtype=dtype))

        sigma = math.sqrt(2/(num_embeddings+embedding_dim))
        nn.init.trunc_normal_(self.E, mean=0, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        return self.E[token_ids, :]


class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):

        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.G = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(1/self.d_model *
                         torch.sum(x**2, dim=-1, keepdim=True)+self.eps)

        return (x/rms * self.G).to(in_dtype)


class SwiGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = nn.Parameter(torch.empty(
            d_ff, d_model, device=device, dtype=dtype))
        self.W2 = nn.Parameter(torch.empty(
            d_model, d_ff, device=device, dtype=dtype))
        self.W3 = nn.Parameter(torch.empty(
            d_ff, d_model, device=device, dtype=dtype))

        sigma = math.sqrt(2/(d_ff + d_model))
        nn.init.trunc_normal_(self.W1, mean=0, std=sigma,
                              a=-3*sigma, b=3*sigma)
        nn.init.trunc_normal_(self.W2, mean=0, std=sigma,
                              a=-3*sigma, b=3*sigma)
        nn.init.trunc_normal_(self.W3, mean=0, std=sigma,
                              a=-3*sigma, b=3*sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        w1x = x @ self.W1.transpose(-1, -2)
        w3x = x @ self.W3.transpose(-1, -2)
        silu = w1x * self.sigmoid(w1x)
        return (silu * w3x) @ self.W2.transpose(-1, -2)

    def sigmoid(self, x: torch.Tensor) -> torch.Tensor:

        neg_abs_x = -x.abs()
        exp_neg_abs_x = torch.exp(neg_abs_x)
        pos_sigmoid = 1/(1+exp_neg_abs_x)

        return torch.where(x >= 0, pos_sigmoid, 1-pos_sigmoid)


class RoPE(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):

        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        cos, sin = self.precompute_rope(self.max_seq_len, self.d_k)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:

        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        else:
            token_positions = token_positions.flatten()

        cos = self.cos[..., token_positions, :]
        sin = self.sin[..., token_positions, :]

        if x.dim() == 3:
            cos = cos.squeeze(0).squeeze(0)
            sin = sin.squeeze(0).squeeze(0)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        return torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)

    def precompute_rope(self, seq_len: int, d: int) -> tuple[torch.Tensor]:

        # precompute rope for (seq_len, d)

        # pair index
        k = torch.arange(0, d//2, device=self.device, dtype=self.dtype)
        base_frequency = self.theta ** (-2*k/d)
        # position index
        i = torch.arange(0, seq_len, device=self.device, dtype=self.dtype)
        # cartesian product i x k
        angle = i[:, None] * base_frequency[None, :]

        # shape seq_len, dim/2
        cos = torch.cos(angle)
        sin = torch.sin(angle)

        # add batch and head axis for later broadcasting:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        return cos.to(self.device), sin.to(self.device)


def softmax(x: torch.Tensor, d: int = -1, temperature=1) -> torch.Tensor:

    dtype = x.dtype
    x = x.to(torch.float32)

    x = torch.exp((x-torch.max(x, dim=d, keepdim=True).values)/temperature)

    return (x/torch.sum(x, dim=d, keepdim=True)).to(dtype)


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, rope: nn.Module = None, device=None, dtype=None, max_seq_len: int = 512, mask: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.use_precomputed_mask = mask
        if rope is not None:
            self.rope = rope
        self.WQ = nn.Parameter(torch.empty(
            d_model, d_model, device=device, dtype=dtype))
        self.WK = nn.Parameter(torch.empty(
            d_model, d_model, device=device, dtype=dtype))
        self.WV = nn.Parameter(torch.empty(
            d_model, d_model, device=device, dtype=dtype))
        self.WO = nn.Parameter(torch.empty(
            d_model, d_model, device=device, dtype=dtype))

        sigma = math.sqrt(2/(d_model + d_model))
        nn.init.trunc_normal_(
            self.WQ, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)
        nn.init.trunc_normal_(
            self.WK, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)
        nn.init.trunc_normal_(
            self.WV, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)
        nn.init.trunc_normal_(
            self.WO, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)

        if mask:
            # precompute causal mask (additive, with -inf for masked positions)
            causal_mask = torch.zeros(
                max_seq_len, max_seq_len, device=device, dtype=dtype)
            causal_mask.masked_fill_(torch.triu(torch.ones(
                max_seq_len, max_seq_len, device=device), diagonal=1).bool(), float('-inf'))
            self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: int = None):

        batch, seq_len, _ = x.shape

        # Apply projection matrix and split d_model into n_heads, d_head + move n_heads to the front as batch dimension
        Q = (x @ self.WQ.T).view(batch, seq_len,
                                 self.num_heads, self.d_head).transpose(-2, -3)
        K = (x @ self.WK.T).view(batch, seq_len,
                                 self.num_heads, self.d_head).transpose(-2, -3)
        V = (x @ self.WV.T).view(batch, seq_len,
                                 self.num_heads, self.d_head).transpose(-2, -3)

        # Apply RoPE
        if hasattr(self, "rope"):
            Q = self.rope.forward(Q, token_positions)
            K = self.rope.forward(K, token_positions)

        # Calculate Attention
        if self.use_precomputed_mask:
            output = self.scaled_dot_product_attention(
                Q, K, V, precomputed_mask=self.causal_mask[:seq_len, :seq_len])
        else:
            mask = torch.tril(torch.ones(seq_len, seq_len,
                              device=x.device, dtype=torch.bool))
            output = self.scaled_dot_product_attention(Q, K, V, mask=mask)

        # Concatenate heads
        output = output.transpose(1, 2).reshape(
            batch, seq_len, self.d_model)
        # Apply output projection
        return output @ self.WO.transpose(-2, -1)

    @staticmethod
    def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None, precomputed_mask: torch.Tensor = None) -> torch.Tensor:

        d_k = K.shape[-1]
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)

        if precomputed_mask is not None:
            scores = scores + precomputed_mask
        elif mask is not None:
            scores = scores.masked_fill(~mask, -float('inf'))

        return softmax(scores) @ V


class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, num_heads: int, weights: dict[str, torch.Tensor], rope, device=None, dtype=None, eps=1e-5, norm=True, max_seq_len: int = 512):

        super().__init__()

        self.device = device

        self.rope = rope

        if norm:

            self.rmsnorm1 = RMSNorm(d_model, eps, device, dtype)

            self.rmsnorm2 = RMSNorm(d_model, eps, device, dtype)

        self.mha = MultiHeadSelfAttention(
            d_model, num_heads, rope=rope, max_seq_len=max_seq_len, device=device, dtype=dtype)

        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

        if weights is not None:
            self.ffn.load_state_dict(
                {"W1": weights["ffn.w1.weight"], "W2": weights["ffn.w2.weight"], "W3": weights["ffn.w3.weight"]})
            self.mha.load_state_dict({"WQ": weights["attn.q_proj.weight"], "WK": weights["attn.k_proj.weight"],
                                      "WV": weights["attn.v_proj.weight"], "WO": weights["attn.output_proj.weight"]})
            if norm:
                self.rmsnorm1.load_state_dict({"G": weights["ln1.weight"]})
                self.rmsnorm2.load_state_dict({"G": weights["ln2.weight"]})

    def forward(self, x, norm=True):

        if norm:
            # Sublayer 1:
            x = x + self.mha(self.rmsnorm1(x),
                             torch.arange(x.shape[-2], device=self.device))

            # Sublayer 2:
            return x + self.ffn(self.rmsnorm2(x))

        x = x + self.mha(x, torch.arange(x.shape[-2], device=self.device))
        return x + self.ffn(x)


class Transformer(nn.Module):

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, weights, device, dtype, norm=True, rope=True):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.d_model = d_model
        self.context_length = context_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.d_ff = d_ff

        self.embedding = Embedding(vocab_size, d_model, device, dtype)

        if rope is True:
            self.rope = RoPE(rope_theta, d_model/num_heads,
                             context_length, device, dtype)
        else:
            self.rope = None

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                d_ff=d_ff,
                num_heads=num_heads,
                weights=None if weights is None else {
                    k.replace(f"layers.{i}.", ""): v
                    for k, v in weights.items()
                    if k.startswith(f"layers.{i}.")
                },
                rope=self.rope,
                device=device,
                dtype=dtype,
                eps=1e-5,
                norm=norm,
                max_seq_len=context_length
            )
            for i in range(num_layers)
        ])

        if norm is True:
            self.ln_final = RMSNorm(d_model=d_model, eps=1e-5,
                                    device=device, dtype=dtype)

        self.output = Linear(d_model, vocab_size, device, dtype)

        if weights is not None:
            self.output.load_state_dict({"W": weights["lm_head.weight"]})
            self.embedding.load_state_dict(
                {"E": weights["token_embeddings.weight"]})
            if norm is True:
                self.ln_final.load_state_dict(
                    {"G": weights["ln_final.weight"]})

    def forward(self, x, norm=True):

        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, norm=norm)

        # output projection onto vocabulary:
        if norm is True:
            return self.output(self.ln_final(x))
        return self.output(x)

    def get_flops(self, verbose=False):

        # FLOP calculations only considering Matmuls and an input sequence of context_length
        # (each matmul (m,n) @ (n,k) is multiply + add -> 2FLOPS * m*n*k

        # 1. Embedding Layer: no matmuls (we simply index the Embedding matrix)
        # 2. RMSNorm: no matmuls
        # 3. FFN:
        # x @ W1 (seq_len, d_model) @ (d_model, d_ff) - same for x @ W3 and y @ W2
        # Complexty:O(d * d_ff *T)
        flops_ffn = 3 * self.d_model * self.d_ff * self.context_length * 2
        # 4. MHA:
        # 4.1 Projections: xWq, xWk, xWv, xWO (assuming d_k = d_V = d_model)
        # (seq_len, d_model) @ (d_model, d_model) Complexity: O(d^2*T)
        flops_attention = 4 * self.d_model ** 2 * self.context_length * 2
        # 4.2 Attention calculations
        # scores: (seq_len, d_model) @ (d_model, seq_len) Complexity: O(T^2*d)
        flops_attention += self.context_length ** 2 * self.d_model * 2
        # scores @ V (seq_len, seq_len) @ (seq_len, d_model) Complexity: O(T^2*d)
        flops_attention += self.context_length ** 2 * self.d_model * 2

        # Multiply by number of layers
        flops = flops_ffn + flops_attention
        flops *= self.num_layers

        # 5. Output Projection : (seq_len, d_model) (d_model, vocab_size)
        flops_output = self.context_length * self.d_model * self.vocab_size * 2
        flops += flops_output

        if verbose:
            print(
                f"The model has a total of {flops/1e9:.2f} GFLOPS for one forward pass")
            print(
                f"Attention: {flops_attention*self.num_layers*100/flops:.2f} % of FLOPS")
            print(f"FFN: {flops_ffn*self.num_layers*100/flops:.2f} % of FLOPS")
            print(
                f"Output Projection: {flops_output*100/flops:.2f} % of FLOPS")
        return flops

    def get_parameters(self, verbose=False, use_muon=False, ddp=False):
        if use_muon:
            muon_params = 0
            adamw_params = 0
            for name, p in self.named_parameters():
                if not p.requires_grad:
                    continue

                if ddp:
                    non_muon = ("module.output.W, module.embedding.E")
                else:
                    non_muon = ("output.W", "embedding.E")

                if len(p.shape) == 2 and name not in non_muon:
                    muon_params += p.numel()
                else:
                    adamw_params += p.numel()

            return (muon_params, adamw_params)

        parameters = sum(p.numel() for p in self.parameters())
        if verbose:
            print(
                f"The model has a total of {parameters/1e6:.2f} M Parameters")
        return parameters

    def get_activation_size(self, verbose=False, flash_attention=False):
        """
        Activation memory for training: what autograd saves for backward pass.
        Reference: https://blog.eleuther.ai/transformer-math/
        """
        bytes_per_elem = torch.tensor([], dtype=self.dtype).element_size()
        T = self.context_length
        d = self.d_model
        d_ff = self.d_ff
        L = self.num_layers
        n_heads = self.num_heads
        V = self.vocab_size

        if flash_attention:

            # 1. RMSNorm:
            attn_norm_input = T * d

            qkv_input = T * d

            # 3. Q, K, V matrices
            qkv_matrices = 3 * T * d

            # L
            l = T

            # 7. Attention output before projection
            attn_out = T * d
            # 8. Output projection input
            out_proj_input = T * d

            attn_per_layer = attn_norm_input + qkv_input + \
                qkv_matrices + attn_out + out_proj_input + l

        else:

            # Attention block
            # 1. RMSNorm:
            attn_norm_input = T * d
            # 2. QKV projection:
            qkv_input = T * d
            # 3. Q, K matrices
            qk_matrices = 2 * T * d
            # 4. Attention scores:
            attn_scores = n_heads * T * T
            # 5. Softmax output
            softmax_out = n_heads * T * T
            # 6. V matrix
            v_matrix = T * d
            # 7. Attention output before projection
            attn_out = T * d
            # 8. Output projection input
            out_proj_input = T * d

            attn_per_layer = (attn_norm_input + qkv_input + qk_matrices +
                              attn_scores + softmax_out + v_matrix +
                              attn_out + out_proj_input)

        # ffn(swiglu)
        # 1. RMSNormt
        ffn_norm_input = T * d
        # 2. FFN input
        ffn_input = T * d
        # 3. W1 @ x
        w1x = T * d_ff
        # 4. W3 @ x
        w3x = T * d_ff
        # 5. silu intermediate
        silu_intermediate = T * d_ff
        # 6. silu * w3x
        ffn_hidden = T * d_ff

        ffn_per_layer = (ffn_norm_input + ffn_input + w1x + w3x +
                         silu_intermediate + ffn_hidden)

        # residual connections
        residual_per_layer = 2 * T * d
        per_layer_total = attn_per_layer + ffn_per_layer + residual_per_layer
        # for all layers:
        memory = per_layer_total * L

        # only once:
        # 1. embedding
        memory += T * d
        # 2. Final RMSNorm
        memory += T * d
        # 3. Output projection
        memory += T * d
        # 4. Logits
        memory += T * V

        # Convert to bytes
        memory *= bytes_per_elem

        if verbose:
            print("Activation memory(batch_size=1):")
            print(
                f"  Per layer: {per_layer_total * bytes_per_elem / 1e6:.2f} MB")
            print(
                f"    Attention: {attn_per_layer * bytes_per_elem / 1e6:.2f} MB")
            print(f"    FFN: {ffn_per_layer * bytes_per_elem / 1e6:.2f} MB")
            print(
                f"    Residuals: {residual_per_layer * bytes_per_elem / 1e6:.2f} MB")
            print(
                f"  All {L} layers: {per_layer_total * L * bytes_per_elem / 1e6:.2f} MB")
            print(
                f"  Output/embedding: {(3*T*d + T*V) * bytes_per_elem / 1e6:.2f} MB")
            print(f"Total activations (batch=1): {memory / (1024**3):.3f} GB")

        return memory

    def get_memory(self, verbose=False, use_muon=False):
        if use_muon:

            parameters_muon, parameters_adamw = self.get_parameters(
                verbose=False, use_muon=use_muon)
            memory_muon = parameters_muon * \
                torch.tensor([], dtype=self.dtype).element_size()
            memory_adamw = parameters_adamw * \
                torch.tensor([], dtype=self.dtype).element_size()

            return memory_muon, memory_adamw
        parameters = self.get_parameters()
        memory = parameters * torch.tensor([], dtype=self.dtype).element_size()
        if verbose:
            print(
                f"The model consumes a total memory of {memory/(1024**3):.2f} GB ")
        return memory

    def get_training_flops(self, verbose=False):

        flops = self.get_flops() * 3 + 17 * self.get_parameters()

        if verbose:
            print(
                f"Training the model takes a total of {flops/1e9:.2f} GFLOPS per batch per step")
        return flops

    def get_training_memory(self, verbose=False, batch_size=1, use_muon=False, ddp=False, world_size=None, shard_gradient=False, shard_optimizer=False, mixed_precision=False):
        """
        Total memory for training including activations, weights, gradients, and optimizer states.
        Including empirical overhead factors for memory fragmentation and torch.compile.
        """
        # Memory fragmentation overhead from pytorch allocator
        FRAGMENTATION_FACTOR = 1.3

        activation_mem = self.get_activation_size(
            flash_attention=ddp) * batch_size

        # Mixed precision stores activations in fp16
        if mixed_precision:
            activation_mem /= 2

        if ddp:
            activation_mem /= world_size

        model_mem = self.get_memory()

        if use_muon:
            memory_muon, memory_adamw = self.get_memory(
                verbose=False, use_muon=use_muon)

            # the optimizer needs gradients + momentum for muon (+ velocity) for adamw
            if shard_gradient:
                optimizer_mem = memory_muon * \
                    (2/world_size) + memory_adamw * (3/world_size)
            elif shard_optimizer:
                optimizer_mem = memory_muon * \
                    (1+1/world_size) + memory_adamw * (1+2/world_size)
            else:
                optimizer_mem = memory_muon * 2 + memory_adamw * 3
            base_memory = activation_mem + model_mem + optimizer_mem
        else:
            if shard_gradient:
                optimizer_mem = model_mem * (3/world_size)
            elif shard_optimizer:
                optimizer_mem = model_mem * (1+2/world_size)
            else:
                optimizer_mem = model_mem * 3
            base_memory = activation_mem + optimizer_mem

        total_memory = base_memory * FRAGMENTATION_FACTOR

        if verbose:
            opt_name = "Muon + AdamW" if use_muon else "AdamW"
            per_gpu_batch = batch_size // world_size if ddp and world_size else batch_size
            print(f"Training memory breakdown ({opt_name}):")
            print(f"  Model weights: {model_mem / 1e9:.3f} GB")
            print(f"  Activations (batch={per_gpu_batch} per GPU): {activation_mem / 1e9:.3f} GB")
            print(f"  Optimizer states: {optimizer_mem / 1e9:.3f} GB")
            print(f"  Total Memory per GPU: {total_memory / 1e9:.3f} GB")

        return total_memory

    def get_training_time(self, verbose=False, steps=None, batch_size=None, ddp=False, world_size=None):

        device_type = self.device.type if hasattr(self.device, 'type') else self.device

        # For Macbook Pro M3 with MPS we can assume a realistic throughput of 10-20% of max throughput (7.4 TFLOP/S)
        if device_type == "mps":
            throughput = 1.15 * 1e12

        # For a T4 with Flash Attention and mixed precision:
        if device_type == "cuda":
            throughput = 15 * 1e12

        # With distributed training on T4s - measured ~13 TFLOPS per GPU with DDP + gradient accumulation
        if device_type == "cuda" and ddp:
            throughput = 13 * 1e12 * world_size

        # FLOPS per step of training:
        flops = self.get_training_flops()

        if steps is None:
            # Assuming Chinchillas Law (we want to train on roughly 20 * parameters tokens).
            # More recent papers tend towards 8-9x scaling law

            tokens = 8 * self.get_parameters()

            # For a given context length we can fit len(context_window) tokens per step
            steps = tokens/self.context_length

            chincilla_training_time = flops * steps / throughput

            if verbose:
                print(
                    f"Training the model locally according to Chincillas Law will take roughly {chincilla_training_time / 3600:.2f} hours")
            return chincilla_training_time

        else:
            training_time = flops * batch_size * steps / throughput
            if verbose:
                print(
                    f"Training the model locally will take roughly {training_time / 3600:.2f} hours")
            return training_time


# Calculations for GPT-2 XL:
# transformer = Transformer(50257, 1024, 1600, 48, 25,
#                           6400, 1, None, None, torch.float32)

# for reasonably sized local model:
# transformer = Transformer(32000, 512, 768, 12, 12, 2048,
#                           10000, None, None, torch.float32, True, True)


# transformer.get_parameters(verbose=True)
# transformer.get_memory(verbose=True)
# transformer.get_flops(verbose=True)
# transformer.get_activation_size(verbose=True)
# transformer.get_training_flops(verbose=True)
# transformer.get_training_memory(verbose=True, batch_size=32, use_muon=True)
# transformer.get_training_time(verbose=True)
