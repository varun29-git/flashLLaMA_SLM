import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------------------------------

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
    
    def forward(self, x):
        return self.embedding(x)

# ------------------------------------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        # Llama uses 1e-6
        super().__init__()

        # RMS normalizes scale only, so no beta parameter
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.epsilon = eps
    
    def forward(self, x):
        sqrs = x.square()        
        mean_sqr = sqrs.mean(dim=-1, keepdim=True)

        # Add epsilon to avoid division by zero
        mean_sqr = mean_sqr + self.epsilon
        RMS = mean_sqr.sqrt()

        # Each token normalized by the RMS
        x = x / RMS

        # Scale by gamma
        return self.gamma * x
    
# ------------------------------------------------------------------------------------------------------

class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, 2 * d_ff)
        # No bias in the output layer
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)

    @staticmethod
    def SWiGLU(x):
        # Input split into two parts
        x1, x2 = x.chunk(2, dim=-1)
        # SwiGLU(x) = SiLU(x1) * x2
        return F.silu(x1) * x2

    def forward(self, x):
        x = self.linear1(x)
        x = self.SWiGLU(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

# ------------------------------------------------------------------------------------------------------

class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, *args, **kwargs):
        return x + self.dropout(sublayer(self.norm(x), *args, **kwargs))
    
# ------------------------------------------------------------------------------------------------------


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values

# Compile the function if PyTorch 2.0+ is available
if hasattr(torch, "compile"):
    repeat_kv = torch.compile(repeat_kv)

class GQA_Flash_RoPE(nn.Module):
    
    def __init__(self, d_model, num_q_heads, num_kv_heads, dropout):
        super().__init__()

        assert d_model % num_q_heads == 0, "d_model must be divisible by num_q_heads"
        assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_q_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, self.d_k * num_kv_heads, bias=False)
        self.w_v = nn.Linear(d_model, self.d_k * num_kv_heads, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # RoPE buffers
        self.register_buffer("sin", None, persistent=False)
        self.register_buffer("cos", None, persistent=False)

        # Cache buffers
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)

      
    @staticmethod
    def apply_rope(x, sin, cos):
        
        # Get the last dimension and put the assertion
        d_k = x.shape[-1]
        assert d_k % 2 == 0, "RoPE requires even d_k"

        # If sin and cos are 2D, expand them to match the batch and head dimensions
        if sin.dim() == 2:      # (T, d_k//2)
            sin = sin[None, None, :, :]  # (1, 1, T, d_k//2)
            cos = cos[None, None, :, :]
        
        # Split even and odd
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        # Apply RoPE to even and odd
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_odd  * cos + x_even * sin

        # Combine even and odd
        x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)
        return x_rot
    
    @staticmethod
    def get_rope_sin_cos(T, d_k, device):

        # Assert d_k is even
        assert d_k % 2 == 0

        # Positional encoding
        pos = torch.arange(T, device=device)          # (T,)
        dim = torch.arange(0, d_k, 2, device=device)  # (d_k/2,)
        
        # Inverse frequency
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        angles = pos[:, None] * inv_freq[None, :]     # (T, d_k/2)

        # Sine and cosine
        sin = angles.sin()[None, None, :, :]  # (1, 1, T, d_k/2)
        cos = angles.cos()[None, None, :, :]  # (1, 1, T, d_k/2)

        return sin, cos
    
    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None

    
    def forward(self, x, start_pos=0, use_cache=False):
        
        B, T, d_model = x.shape

        # Project to Q, K, V
        query = self.w_q(x)
        key   = self.w_k(x)
        value = self.w_v(x)
        
        # Reshaping
        query = query.reshape(B, T, self.num_q_heads, self.d_k).transpose(1, 2).contiguous()
        key = key.reshape(B, T, self.num_kv_heads, self.d_k).transpose(1, 2).contiguous()
        value = value.reshape(B, T, self.num_kv_heads, self.d_k).transpose(1, 2).contiguous()

        # RoPE cache handling
        if self.sin is None or self.sin.shape[-2] < start_pos + T:
            sin, cos = self.get_rope_sin_cos(start_pos + T, self.d_k, x.device)
            self.sin = sin.to(x.dtype)
            self.cos = cos.to(x.dtype)

        sin = self.sin[:, :, start_pos:start_pos + T, :].to(device=x.device, dtype=x.dtype)
        cos = self.cos[:, :, start_pos:start_pos + T, :].to(device=x.device, dtype=x.dtype)
        
        # Apply RoPE
        query = self.apply_rope(query, sin, cos)
        key = self.apply_rope(key, sin, cos)
        
        # Cache handling
        if use_cache:
            if self.k_cache is None:
                self.k_cache = key
                self.v_cache = value
            else:
                self.k_cache = torch.cat([self.k_cache, key], dim=2)
                self.v_cache = torch.cat([self.v_cache, value], dim=2)
            
            key, value = self.k_cache, self.v_cache
        
        # Broadcast key and value
        num_groups = self.num_q_heads // self.num_kv_heads
        key, value = repeat_kv(key, value, num_groups, 1)

        # Flash attention with Pytorch
        attn_out = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0, is_causal=True)

        # Reshape and project to output
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, T, self.num_q_heads * self.d_k)

        output = self.w_o(attn_out)
        return output
  
# ------------------------------------------------------------------------------------------------------

        

class DecoderBlock(nn.Module):

    def __init__(self, d_model, attention: GQA_Flash_RoPE , feed_forward: FeedForward, dropout):
        
        super().__init__()
        self.attention = attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout), ResidualConnection(d_model, dropout)])

    def forward(self, x, start_pos=0, use_cache=False):
        x = self.residual_connections[0](x, self.attention, start_pos, use_cache) # GQA + residual
        x = self.residual_connections[1](x, self.feed_forward) # Feed-forward + residual
        return x

# ------------------------------------------------------------------------------------------------------

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList, d_model):
        super().__init__()
        self.layers = layers
        self.norm = RMSNorm(d_model)

    def forward(self, x, start_pos=0, use_cache=False):
        # Pass through all decoder blocks
        for layer in self.layers:
            x = layer(x, start_pos=start_pos, use_cache=use_cache)
        return self.norm(x)

# ------------------------------------------------------------------------------------------------------

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        return self.linear(x)

# ------------------------------------------------------------------------------------------------------

class FlashLLaMA(nn.Module):

    def __init__(self, embedding, decoder, projection):
        super().__init__()
        self.embedding = embedding
        self.decoder = decoder
        self.projection = projection

    def forward(self, x, start_pos=0, use_cache=False):
        x = self.embedding(x)
        x = self.decoder(x, start_pos=start_pos, use_cache=use_cache)
        x = self.projection(x)
        return x
    
    def reset_kv_cache(self):
        for layer in self.decoder.layers:
            layer.attention.reset_cache()


# ------------------------------------------------------------------------------------------------------

def build_llama(vocab_size, d_model=1024, num_layers=12, num_q_heads=8, num_kv_heads=8, d_ff=2048, dropout=0.1):

    # Input embedding
    embedding = InputEmbedding(vocab_size, d_model)

    # Decoder layers
    decoder_layers = []
    for _ in range(num_layers):
        attention = GQA_Flash_RoPE(d_model, num_q_heads, num_kv_heads, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        decoder_layers.append(DecoderBlock(d_model, attention, feed_forward, dropout))

    # Decoder
    decoder = Decoder(nn.ModuleList(decoder_layers), d_model)

    # Projection layer
    projection = ProjectionLayer(d_model, vocab_size)

    # Weight tying
    projection.linear.weight = embedding.embedding.weight

    # Model
    model = FlashLLaMA(embedding, decoder, projection)
    return model

# ------------------------------------------------------------------------------------------------------

