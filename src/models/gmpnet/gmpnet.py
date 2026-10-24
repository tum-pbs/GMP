"""GMPNet: Scalable Mesh-free Aerodynamic Simulations from Raw Geometries using a Transformer-based Surrogate Model

This module contains the implementation of the encoder and decoder blocks used in GMPNet, as well as the complete GMPNet model.
Designed for simulating time-independent PDEs over complex 3D geometries, GMPNet leverages a Transformer-based architecture
to perform simulations using solely inexpensive geometry meshes, eliminating the need for costly surface and volumetric
meshing during inference.
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ModulatedPositionalEmbedding(nn.Module):
    """Embedding layer that applies a modulated sine-cosine positional embedding to the spatial positions. This means 
    that the sine-cosine functions are shifted and scaled based on learned parameters from an MLP for each position.
    This allows the model to adaptively adjust the positional embeddings based on the data to emphasize or suppress
    high-frequency variations.

    The implementation follows the original Transformer positional embedding (https://arxiv.org/abs/1706.03762) and
    the class is based on the PositionalEncoding class from the PyTorch tutorial 'Language Modeling with nn.Transformer and torchtext'.
    
    Args:
        dim: Dimensionality of the embedded positions.
        spatial_dim: The spatial dimensionality of the positions (e.g., 2 for 2D positions, 3 for 3D positions). Defaults to 3.
        max_seq_length: Max sequence length. Defaults to 10000 as suggested in the original Transformer paper.
    """

    def __init__(self, dim, spatial_dim=3, max_seq_length=10000):
        super().__init__()
        self.dim = dim
        self.spatial_dim = spatial_dim
        
        # Compute dimensions per spatial dimension
        max_dim_per_spatial_dim = dim // spatial_dim
        dim_per_spatial_dim = max_dim_per_spatial_dim & ~1 # This is equal to (max_dim_per_spatial_dim // 2) * 2
        self.dim_per_spatial_dim = dim_per_spatial_dim
        
        # Compute the total padding
        self.total_padding = dim - (dim_per_spatial_dim * spatial_dim)
        self.register_buffer("padding", torch.zeros(1, 1, self.total_padding))
        
        # Compute the div_term for sine-cosine embedding
        div_term = torch.exp(torch.arange(0, dim_per_spatial_dim, 2) * (-math.log(max_seq_length) / dim_per_spatial_dim))
        self.register_buffer("div_term", div_term)
        
        # Modulation MLP
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim_per_spatial_dim * spatial_dim * 2))

    def compute_embedding(self, pos, shift_sin=None, scale_sin=None, shift_cos=None, scale_cos=None):
        # Following UPT (https://arxiv.org/abs/2402.12365) and compute positional embeddings in float32 to avoid numerical instabilities
        with torch.autocast(device_type=str(pos.device).split(":")[0], enabled=False):
            pos = pos.float()
            sin_cos_arg = pos[..., None] @ self.div_term[None, ...]
            
            embedding = torch.zeros((*sin_cos_arg.shape[:-1], self.dim_per_spatial_dim), device=sin_cos_arg.device, dtype=sin_cos_arg.dtype)
            # Apply shift and scale to embedding if provided
            if shift_sin is not None and scale_sin is not None and shift_cos is not None and scale_cos is not None:
                embedding[..., 0::2] = scale_sin * torch.sin(sin_cos_arg + shift_sin)
                embedding[..., 1::2] = scale_cos * torch.cos(sin_cos_arg + shift_cos)
            else:
                embedding[..., 0::2] = torch.sin(sin_cos_arg)
                embedding[..., 1::2] = torch.cos(sin_cos_arg)
            
        # Rearrange spatial dimensions
        embedding = rearrange(embedding, "b n spatial_dim d -> b n (spatial_dim d)")
        
        # Apply padding if necessary
        if self.total_padding > 0: embedding = torch.concat([embedding, self.padding.expand(*embedding.shape[:-1], -1)], dim=-1)
        
        return embedding
        
    def forward(self, pos):
        """Embeds the positions, normalized to [0, max_seq_length], using modulated sine-cosine positional embeddings.

        Args:
            pos: Normalized positions with shape (batch size, number points, spatial_dim).

        Returns:
            Embedded positions with shape (batch size, number points, dim).
        """
        initial_embedding = self.compute_embedding(pos)
        
        # Apply modulation MLP for shift and scaling
        shift_scale = self.mlp(initial_embedding)
        shift_sin, scale_sin, shift_cos, scale_cos = torch.unbind(rearrange(shift_scale, "b n (d shift_scale spatial_dim) -> b n spatial_dim d shift_scale", shift_scale=4, spatial_dim=self.spatial_dim), -1)
        
        embedding = self.compute_embedding(pos, shift_sin=shift_sin, scale_sin=scale_sin, shift_cos=shift_cos, scale_cos=scale_cos)
        
        return embedding


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE; https://arxiv.org/abs/2104.09864) for spatial positions.
    
    Args:
        dim: Dimensionality of the features to be embedded.
        spatial_dim: The spatial dimensionality of the positions (e.g., 2 for 2D positions, 3 for 3D positions). Defaults to 3.
        max_seq_length: Max sequence length. Defaults to 10000 as suggested in the original RoPE paper.
    """
    
    def __init__(self, dim, spatial_dim, max_seq_length=10000.0):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for rotary embeddings"
        
        self.dim = dim
        self.spatial_dim = spatial_dim
        
        # Compute dimensions per spatial dimension
        max_dim_per_spatial_dim = dim // spatial_dim
        dim_per_spatial_dim = max_dim_per_spatial_dim & ~1 # This is equal to (max_dim_per_spatial_dim // 2) * 2
        
        # Compute the padding
        self.total_padding = dim - (dim_per_spatial_dim * spatial_dim)
        self.register_buffer("padding", torch.zeros(1, 1, self.total_padding // 2))
        
        # Compute the div_term for sine-cosine embedding
        div_term = torch.exp(torch.arange(0, dim_per_spatial_dim, 2) * (-math.log(max_seq_length) / dim_per_spatial_dim))
        self.register_buffer("div_term", div_term)

    def forward(self, x, pos):
        """Applies RoPE to the features x based on the positions pos.
        
        Args:
            x: Features to apply RoPE to with shape (batch size, number points, dim).
            pos: Normalized positions with shape (batch size, number points, spatial_dim).
            
        Returns:
            Features with RoPE applied to with shape (batch size, number points, dim).
        """
        # Following UPT (https://arxiv.org/abs/2402.12365) and compute positional embeddings in float32 to avoid numerical instabilities
        with torch.autocast(device_type=str(pos.device).split(":")[0], enabled=False):
            pos = pos.float()
            theta = pos[..., None] @ self.div_term[None, ...]
        
        theta = rearrange(theta, "b n spatial_dim d -> b n (spatial_dim d)")
        
        # Add padding
        theta = torch.concat([theta, self.padding.expand(*theta.shape[:-1], -1)], dim=-1)
        
        # Apply rotation matrix in complex space following Llama 3 implementation
        # (https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/model.py#L65)
        rotation = torch.polar(torch.ones_like(theta), theta)
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        embedded = torch.view_as_real(x_complex * rotation[:, None, ...]).flatten(3)
        
        return embedded.type_as(x)
    

class CrossAttention(nn.Module):
    """Computes multi-head cross-attention (https://arxiv.org/abs/1706.03762) between the query and key/value sequences. It
    optionally applies Rotary Positional Embedding (RoPE; https://arxiv.org/abs/2104.09864) to both the query and key features.

    Args:
        dim: Dimensionality of the query and key/value features.
        num_heads: Number of attention heads. Defaults to 8.
        spatial_dim: Number of spatial dimensions for RoPE. Defaults to 3.
        dropout: Dropout rate. Defaults to 0.1.
    """

    def __init__(self, dim, num_heads=8, spatial_dim=3, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Pre-layer normalization
        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6)
        
        # Projections
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        
        # RoPE
        self.rope = RotaryPositionalEmbedding(dim=dim // num_heads, spatial_dim=spatial_dim)
        
        self.dropout = dropout

    def forward(self, q, kv, q_pos=None, kv_pos=None):
        """Applies pre-norm and computes cross-attention between q and kv.

        Args:
            q: Queries with shape (batch size, number query tokens, dim).
            kv: Key/value with shape (batch size, number key/value tokens, dim).
            q_pos (optional): Positions for the queries with shape (batch size, number query tokens, spatial_dim).
            kv_pos (optional): Positions for the key/value with shape (batch size, num key/value tokens, spatial_dim).

        Returns:
            Updated queries that attend to kv with shape (batch size, number query tokens, dim).
        """
        # Check for all-zero embeddings in kv
        mask = (kv.abs().sum(dim=-1) == 0)  # Shape: (batch_size, num_kv_points)
        if mask.any():
            # Expand mask for multi-head attention
            mask = mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, num_kv_points)
        else:
            mask = None
            
        # Apply layer normalization
        q = self.norm_q(q)
        kv = self.norm_kv(kv)
        
        # Linear projections
        q = self.q(q)
        kv = self.kv(kv)
            
        # Split heads and keys/values
        q_heads = rearrange(q, "b q (h d) -> b h q d", h=self.num_heads, d=self.head_dim)
        k_heads, v_heads = torch.tensor_split(rearrange(kv, "b kv (h d) -> b h kv d", h=2*self.num_heads, d=self.head_dim), 2, dim=1)
        
        # Apply RoPE if positions are provided
        if q_pos is not None and kv_pos is not None:
            q_heads = self.rope(q_heads, q_pos)
            k_heads = self.rope(k_heads, kv_pos)
            
        # Compute attention using PyTorch's scaled_dot_product_attention
        x = F.scaled_dot_product_attention(q_heads, k_heads, v_heads, dropout_p=(self.dropout if self.training else 0.0), attn_mask=mask)
        
        # Merge heads and output projection
        x = rearrange(x, "b h q d -> b q (h d)")
        x = self.out_proj(x)
        
        return x


class Modulator(nn.Module):
    """Modulator module for FiLM-like modulation (https://github.com/ethanjperez/film) of features based on conditioning parameters.
    
    Args:
        dim: Dimensionality of the features to be modulated.
        cond_dim: Dimensionality of the conditioning parameters.
        hidden_dim: Dimensionality of the hidden layer in the modulation MLP. Default is 128.
        use_residual: If true, use residual connection in modulation. Default is False.
    """
    
    def __init__(self, dim, cond_dim, hidden_dim=128, use_residual=False):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cond_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim * 2))
        self.use_residual = use_residual

    def forward(self, x, params):
        """Modulates the features x based on the conditioning parameters params.
        
        Args:
            x: Features to be modulated with shape (batch size, number points, dim).
            params: Conditioning parameters with shape (batch size, cond_dim).
        
        Returns:
            Modulated features with shape (batch size, number points, dim).
        """
        if params is not None and params.ndim == 1:
            params = params.unsqueeze(-1)
        scale, shift = torch.tensor_split(self.mlp(params), 2, dim=-1)
        scale, shift = scale.unsqueeze(1), shift.unsqueeze(1)
        x = scale * x + shift if not self.use_residual else x + scale * x + shift
        return x


class SimulationParamModulatedMLP(nn.Module):
    """MLP with FiLM-like modulation (https://github.com/ethanjperez/film) based on simulation parameters.
    
    Args:
        dim: Dimensionality of the input and output features.
        hidden_dim: Dimensionality of the hidden layer of the MLP.
        cond_dim: Dimensionality of the conditioning parameters.
        dropout: Dropout rate. Default is 0.1.
        use_residual: If true, use residual connection in modulation. Default is False.
    """

    def __init__(self, dim, hidden_dim, cond_dim, dropout=0.1, use_residual=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.non_linearity = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.modulator = Modulator(hidden_dim, cond_dim, use_residual=use_residual)
        

    def forward(self, x, params):        
        """Processes the features x with an MLP, modulated based on the conditioning parameters params.
        
        Args:
            x: Features with shape (batch size, number points, dim).
            params: Conditioning parameters with shape (batch size, cond_dim).
        
        Returns:
            Processed features with shape (batch size, number points, dim).
        """
        x = self.modulator(self.non_linearity(self.linear1(self.norm(x))), params)
        x = self.linear2(x)
        
        return x


class PlainMLP(nn.Module):
    """Plain multi-layer perceptron (MLP) **without** modulation
    
    Args:
        dim: Dimensionality of the input and output features.
        hidden_dim: Dimensionality of the hidden layer of the MLP.
        dropout: Dropout rate. Default is 0.1.
    """

    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.non_linearity = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, dim)
        

    def forward(self, x, params):
        """Processes the features x with an MLP without modulation.
            
        Args:
            x: Features with shape (batch size, number points, dim).
            params: Conditioning parameters will be ignored.
        
        Returns:
            Processed features with shape (batch size, number points, dim).
        """
        x = self.linear2(self.non_linearity(self.linear1(self.norm(x))))
        
        return x


class EncoderBlock(nn.Module):
    """The encoder block updates the latent geometry by attending to a cross-attended version of itself. This means that
    first, the latent geometry attends to a subsampled version of the input geometry to integrate geometric information,
    and then it attends to this cross-attended version of itself to refine the latent representation.

    Args:
        dim: Dimensionality of the features.
        num_heads: Number of attention heads. Defaults to 8.
        dropout: Dropout rate. Defaults to 0.1.
        spatial_dim: Number of spatial dimensions for RoPE. Defaults to 3.
        cond_dim: Dimensionality of the conditioning parameters for the MLP. Defaults to 2.
    """

    def __init__(self, dim, num_heads=8, dropout=0.1, spatial_dim=3, cond_dim=2, shared_attn_geo=None, shared_attn_cross=None, shared_mlp=None):
        super().__init__()
        self.geo_attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim) if shared_attn_geo is None else shared_attn_geo
        self.cross_attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim) if shared_attn_cross is None else shared_attn_cross
        self.attn_dropout = nn.Dropout(dropout)
        
        # Pointwise MLP
        if cond_dim > 0:
            self.mlp = SimulationParamModulatedMLP(dim=dim, hidden_dim=dim * 4, cond_dim=cond_dim, dropout=dropout) if shared_mlp is None else shared_mlp
        else:
            self.mlp = PlainMLP(dim=dim, hidden_dim=dim * 4, dropout=dropout)
       
    def forward(self, latent_geometry, subsampled_geometry, params, latent_geometry_pos=None, subsampled_geometry_pos=None):
        """Updates the latent geometry by attending to a cross-attended version of itself that first attends to a subsampled
        version of the input geometry.

        Args:
            latent_geometry: Latent geometry with shape (batch size, number latent points, dim).
            subsampled_geometry: Subsampled input geometry with shape (batch size, number subsampled points, dim).
            params: Conditioning parameters with shape (batch size, cond_dim).
            latent_geometry_pos (optional): Positions of the latent geometry for the positional embeddings with shape (batch size, number latent points, spatial_dim). Defaults to None.
            subsampled_geometry_pos (optional): Positions of the subsampled input geometry for the positional embeddings with shape (batch size, number subsampled points, spatial_dim) . Defaults to None.

        Returns:
            tuple: A tuple containing:
                - Updated latent geometry with shape (batch size, number latent points, dim).
                - Latent geometry after geometry cross-attention and before cross-attention and MLP with shape (batch size, number latent points, dim).
        """
        # First cross-attention with the subsampled geometry
        latent_geometry_cross = latent_geometry + self.attn_dropout(self.geo_attn(q=latent_geometry, kv=subsampled_geometry, q_pos=latent_geometry_pos, kv_pos=subsampled_geometry_pos))

        # Update the initial latent geometry by attending to the cross-attended version of itself
        latent_geometry_self = latent_geometry + self.attn_dropout(self.cross_attn(q=latent_geometry, kv=latent_geometry_cross, q_pos=latent_geometry_pos, kv_pos=latent_geometry_pos))
        
        # Pointwise MLP
        latent_geometry_mlp = latent_geometry_self + self.mlp(latent_geometry_self, params)
        
        return latent_geometry_mlp, latent_geometry_cross


class DecoderBlock(nn.Module):
    """The decoder block attends to the latent geometry of the corresponding encoder block to produce predictions
    of physical quantities at query positions.

    Args:
        dim: Dimensionality of the features.
        num_heads: Number of attention heads. Defaults to 8.
        dropout: Dropout rate. Defaults to 0.1.
        spatial_dim: Number of spatial dimensions for RoPE. Defaults to 3.
        cond_dim: Dimensionality of the conditioning parameters for the MLP. Defaults to 2.
        shared_attn: Shared cross-attention module from the corresponding encoder block. Defaults to None.
        shared_mlp: Shared MLP module from the corresponding encoder block. Defaults to None.
    """

    def __init__(self, dim, num_heads=8, dropout=0.1, spatial_dim=3, cond_dim=2, shared_attn=None, shared_mlp=None):
        super().__init__()
        self.attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim) if shared_attn is None else shared_attn
        self.attn_dropout = nn.Dropout(dropout)
        
        # Pointwise MLP
        if cond_dim > 0:
            self.mlp = SimulationParamModulatedMLP(dim=dim, hidden_dim=dim * 4, cond_dim=cond_dim, dropout=dropout) if shared_mlp is None else shared_mlp
        else:
            self.mlp = PlainMLP(dim=dim, hidden_dim=dim * 4, dropout=dropout) if shared_mlp is None else shared_mlp
       
    def forward(self, queries, latent_geometry, params, queries_pos=None, latent_geometry_pos=None):
        """Updates the queries by attending to the latent geometry of the corresponding encoder block.

        Args:
            queries: Features of the query positions with shape (batch size, number query points, dim).
            latent_geometry: Latent geometry with shape (batch size, number latent points, dim).
            params: Conditioning parameters with shape (batch size, cond_dim).
            queries_pos (optional): Positions of the query positions for the positional embeddings with shape (batch size, number query points, spatial_dim). Defaults to None.
            latent_geometry_pos (optional): Positions of the latent geometry for the positional embeddings with shape (batch size, number latent points, spatial_dim). Defaults to None.
        
        Returns:
            Updated queries with shape (batch size, number query points, dim).
        """
        # Cross-attention with the latent geometry
        queries = queries + self.attn_dropout(self.attn(q=queries, kv=latent_geometry, q_pos=queries_pos, kv_pos=latent_geometry_pos))
        
        # Pointwise MLP
        queries = queries + self.mlp(queries, params)
        
        return queries


def sample_geometry(geometry, num_samples):
    """Samples points from the input geometry.

    Args:
        geometry: Input geometry with shape (batch size, number points, spatial_dim).
        num_samples: Number of points to sample.
    
    Returns:
        tuple: (sampled_geometry, idx) sampled geometry points and their indices.
    """
    idx = torch.randperm(geometry.shape[1], device=geometry.device)[:num_samples]
    sampled_geometry = geometry[:, idx, :]
    return sampled_geometry, idx

   
class LatentCoAttention(nn.Module):
    """Bidirectional cross-modal latent co-attention between surface and volume latents."""
    def __init__(self, dim, num_heads=8, spatial_dim=3, dropout=0.1):
        super().__init__()
        self.coattn_v2s = CrossAttention(dim=dim, num_heads=num_heads, spatial_dim=spatial_dim, dropout=dropout)
        self.coattn_s2v = CrossAttention(dim=dim, num_heads=num_heads, spatial_dim=spatial_dim, dropout=dropout)
        self.norm_v = nn.LayerNorm(dim, eps=1e-6)
        self.norm_s = nn.LayerNorm(dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tokens_s, tokens_v, pos_s, pos_v):
        if tokens_s is None or tokens_v is None:
            return tokens_s, tokens_v
        # Surface latents attend to volume latents
        s_updated = tokens_s + self.dropout(
            self.coattn_s2v(q=self.norm_s(tokens_s), kv=tokens_v, q_pos=pos_s, kv_pos=pos_v)
        )
        # Volume latents attend to surface latents
        v_updated = tokens_v + self.dropout(
            self.coattn_v2s(q=self.norm_v(tokens_v), kv=tokens_s, q_pos=pos_v, kv_pos=pos_s)
        )
        return s_updated, v_updated


class GMPNet(nn.Module):
    """GMPNet model for simulating time-independent PDEs over complex 3D geometries.
    
    Args:
        spatial_dim: Number of spatial dimensions. Default is 3.
        surface_channels_input: Number of input channels for surface physics. Default is 1.
        volume_channels_input: Number of input channels for volume physics. Default is 3.
        surface_channels: Number of output channels for surface predictions. Default is 1.
        volume_channels: Number of output channels for volume predictions. Default is 3.
        parameter_channels: Number of conditioning parameter channels. Default is 2.
        latent_dim: Dimensionality of the latent representations. Default is 256.
        latent_geometry_points: Number of points of the latent geometry. Default is 4096.
        subsampled_geometry_points: Number of points in the subsampled geometry for geometry cross-attention. Default is 16384.
        num_encoder_decoder_blocks: Number of encoder-decoder blocks. Default is 8.
        num_heads: Number of attention heads. Default is 8.
        pos_scale_factor: Scaling factor for the positions to use more/less of the dynamic range of the positional embedding. Default is 1000.
        dropout: Dropout rate. Default is 0.0.
        use_latent_coattention: Whether to apply bidirectional cross-modal co-attention between surface and volume latents before decoding. Default is False.
    """

    def __init__(self, spatial_dim=3,
                 surface_channels_input=1,
                 volume_channels_input=3,
                 surface_channels=1,
                 volume_channels=3,
                 parameter_channels=2,
                 latent_dim=256,
                 latent_geometry_points=4096,
                 subsampled_geometry_points=16384,
                 num_encoder_decoder_blocks=8, 
                 num_heads=8,
                 pos_scale_factor=1000,
                 dropout=0.0,
                 use_latent_coattention=False,
                 pretrained=False,
                 path_pretrained_encoder_weights=None,
                 path_pretrained_decoder_weights=None,
                 module_encoder_prefixes=None,
                 module_decoder_prefixes=None,
                 exclude_prefixes=None):
        super(GMPNet, self).__init__()
        
        self.spatial_dim = spatial_dim
        self.surface_channels_input = surface_channels_input
        self.volume_channels_input = volume_channels_input
        self.surface_channels = surface_channels
        self.volume_channels = volume_channels
        self.num_geo = latent_geometry_points
        self.subsampled_geometry_points = subsampled_geometry_points
        self.pos_scale_factor = pos_scale_factor
        self.latent_dim = latent_dim
        self.pos_encoder = ModulatedPositionalEmbedding(latent_dim, spatial_dim)
        self.use_latent_coattention = use_latent_coattention

        # Encoder and decoder blocks
        self.encoder_blocks_gs = nn.ModuleList([EncoderBlock(dim=latent_dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim, cond_dim=parameter_channels) for i in range(num_encoder_decoder_blocks)])
        self.encoder_blocks_ps = nn.ModuleList([EncoderBlock(dim=latent_dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim, cond_dim=parameter_channels) for i in range(num_encoder_decoder_blocks)])
        self.encoder_blocks_pv = nn.ModuleList([EncoderBlock(dim=latent_dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim, cond_dim=parameter_channels) for i in range(num_encoder_decoder_blocks)])
        self.decoder_blocks_gs = nn.ModuleList([DecoderBlock(dim=latent_dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim, cond_dim=parameter_channels, shared_attn=self.encoder_blocks_gs[i].cross_attn, shared_mlp=self.encoder_blocks_gs[i].mlp) for i in range(num_encoder_decoder_blocks)])
        self.decoder_blocks_ps = nn.ModuleList([DecoderBlock(dim=latent_dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim, cond_dim=parameter_channels, shared_attn=self.encoder_blocks_ps[i].cross_attn, shared_mlp=self.encoder_blocks_ps[i].mlp) for i in range(num_encoder_decoder_blocks)])
        self.decoder_blocks_pv = nn.ModuleList([DecoderBlock(dim=latent_dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim, cond_dim=parameter_channels, shared_attn=self.encoder_blocks_pv[i].cross_attn, shared_mlp=self.encoder_blocks_pv[i].mlp) for i in range(num_encoder_decoder_blocks)])

        # First physics MLP to project the input physics features to the latent dimension
        self.physics_mlp_surface = nn.Sequential(nn.Linear(self.surface_channels_input, 128), nn.GELU(), nn.Linear(128, latent_dim))
        self.physics_mlp_volume = nn.Sequential(nn.Linear(self.volume_channels_input, 128), nn.GELU(), nn.Linear(128, latent_dim))
        
        # Final MLP
        self.mlp_final = nn.Sequential(nn.Linear(latent_dim, 128), nn.GELU(),
                                 nn.Linear(128, 64), nn.GELU(),
                                 nn.Linear(64, self.surface_channels + self.volume_channels))
        
        # Router mlp_gate configuration (channel router)
        self.mlp_gate = nn.Sequential(nn.Linear(latent_dim, 128), nn.GELU(),
                                     nn.Linear(128, 3 * latent_dim), nn.Sigmoid())

        # Pre-decoder cross-modal latent co-attention
        if self.use_latent_coattention:
            self.latent_coattn = LatentCoAttention(dim=latent_dim, num_heads=num_heads, spatial_dim=spatial_dim, dropout=dropout)

        if pretrained:
            assert path_pretrained_encoder_weights is not None and path_pretrained_decoder_weights is not None, "Paths to pretrained geometry and physics encoders must be provided when pretrained is True"

            if not os.path.exists(path_pretrained_encoder_weights):
                raise ValueError(f"Pretrained encoder weights not found at {path_pretrained_encoder_weights}.")
            if not os.path.exists(path_pretrained_decoder_weights):
                raise ValueError(f"Pretrained decoder weights not found at {path_pretrained_decoder_weights}.")

            # Load and filter pretrained weights
            if path_pretrained_encoder_weights.endswith(".ckpt"):
                checkpoint_encoder = torch.load(path_pretrained_encoder_weights, map_location="cpu", weights_only=False)
                state_dict_geo = checkpoint_encoder.get("state_dict", checkpoint_encoder)
                state_dict_geo = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict_geo.items()}
                
            elif path_pretrained_encoder_weights.endswith(".pth"):
                state_dict_geo = torch.load(path_pretrained_encoder_weights, map_location="cpu")
                state_dict_geo = state_dict_geo["model_state_dict"] 

            if path_pretrained_decoder_weights.endswith(".ckpt"):
                checkpoints_decoder = torch.load(path_pretrained_decoder_weights, map_location="cpu", weights_only=False)
                state_dict_phy = checkpoints_decoder.get("state_dict", checkpoints_decoder)
                state_dict_phy = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict_phy.items()}

            elif path_pretrained_decoder_weights.endswith(".pth"):
                state_dict_phy = torch.load(path_pretrained_decoder_weights, map_location="cpu")
                state_dict_phy = state_dict_phy["model_state_dict"]

            modules_from_geo = tuple(module_encoder_prefixes)
            modules_from_phy = tuple(module_decoder_prefixes)
            exclude_prefixes = tuple(exclude_prefixes)

            # Filter out excluded keys and mismatched keys
            filtered_state_dict = {}
            for k, v in self.state_dict().items():
                if any(k.startswith(p) for p in exclude_prefixes):
                    print(f"Excluding key: {k}")
                    continue
                elif any(k.startswith(p) for p in modules_from_geo):
                    if k in state_dict_geo:
                        if self.state_dict()[k].shape == state_dict_geo[k].shape:
                            filtered_state_dict[k] = state_dict_geo[k]
                            print(f"Including key from geo checkpoint: {k}")
                        else:
                            print(f"Skipping key {k} from geo checkpoint due to shape mismatch (checkpoint: {state_dict_geo[k].shape}, model: {self.state_dict()[k].shape})")
                elif any(k.startswith(p) for p in modules_from_phy):
                    if k in state_dict_phy:
                        if self.state_dict()[k].shape == state_dict_phy[k].shape:
                            filtered_state_dict[k] = state_dict_phy[k]
                            print(f"Including key from phy checkpoint: {k}")
                        else:
                            print(f"Skipping key {k} from phy checkpoint due to shape mismatch (checkpoint: {state_dict_phy[k].shape}, model: {self.state_dict()[k].shape})")
                else:
                    print(f"Key {k} not found in either checkpoint, initializing randomly.")

            missing, unexpected = self.load_state_dict(filtered_state_dict, strict=False)
            self.missing_keys = missing

            print(f"Loaded {len(filtered_state_dict)} tensors. "
                  f"The following layers will be trained from SCRATCH: {missing}")
            
    def initialize_weights(self):
        self.apply(self._init_weights)

    # Weight initialization from Transolver
    # (https://github.com/thuml/Transolver/blob/a11be9c4f7db1885e4b08c68432bc31799492ec9/Car-Design-ShapeNetCar/models/Transolver.py#L168)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def count_active_parameters(self, has_geo_surface=None, has_physics_surface=None, has_physics_volume=None):
        """Calculates the exact number of unique active parameters based on active branches.
        
        Args:
            has_geo_surface: Whether geometry surface branch is used (defaults to True).
            has_physics_surface: Whether physics surface branch is used (defaults to surface_channels_input > 0).
            has_physics_volume: Whether physics volume branch is used (defaults to volume_channels_input > 0).
            
        Returns:
            dict with 'active', 'total', and 'pct' keys.
        """
        if has_physics_surface is None:
            has_physics_surface = (self.surface_channels_input > 0)
        if has_physics_volume is None:
            has_physics_volume = (self.volume_channels_input > 0)
        if has_geo_surface is None:
            if self.surface_channels > 0 or self.surface_channels_input > 0:
                has_geo_surface = True
            elif self.volume_channels > 0 or self.volume_channels_input > 0:
                has_geo_surface = False
            else:
                has_geo_surface = True

        active_modules = [self.pos_encoder, self.mlp_final, self.mlp_gate]
        if has_geo_surface:
            active_modules.extend(self.encoder_blocks_gs)
            active_modules.extend(self.decoder_blocks_gs)
        if has_physics_surface:
            active_modules.append(self.physics_mlp_surface)
            active_modules.extend(self.encoder_blocks_ps)
            active_modules.extend(self.decoder_blocks_ps)
        if has_physics_volume:
            active_modules.append(self.physics_mlp_volume)
            active_modules.extend(self.encoder_blocks_pv)
            active_modules.extend(self.decoder_blocks_pv)
        if self.use_latent_coattention and (has_geo_surface or has_physics_surface) and has_physics_volume:
            if hasattr(self, "latent_coattn"):
                active_modules.append(self.latent_coattn)

        # Unique active parameters
        active_params = set()
        for mod in active_modules:
            active_params.update(mod.parameters())

        total_params = sum(p.numel() for p in self.parameters())
        active_numel = sum(p.numel() for p in active_params)
        return {
            "active": active_numel,
            "total": total_params,
            "pct": (active_numel / total_params * 100.0) if total_params > 0 else 0.0,
            "has_geo_surface": has_geo_surface,
            "has_physics_surface": has_physics_surface,
            "has_physics_volume": has_physics_volume,
        }

    def print_parameter_breakdown(self, has_geo_surface=None, has_physics_surface=None, has_physics_volume=None):
        """Prints a detailed tabular breakdown of active vs idle parameters."""
        stats = self.count_active_parameters(has_geo_surface, has_physics_surface, has_physics_volume)
        idle_params = stats["total"] - stats["active"]
        idle_pct = 100.0 - stats["pct"]
        print("=" * 70)
        print(f"{'GMPNet Parameter Breakdown':^70}")
        print("=" * 70)
        print(f"  Active Branches: Geo Surface={stats['has_geo_surface']}, Physics Surface={stats['has_physics_surface']}, Physics Volume={stats['has_physics_volume']}")
        print("-" * 70)
        print(f"  Active Parameters (forward pass): {stats['active']:,} ({stats['active']/1e6:.3f} M, {stats['pct']:.1f}%)")
        print(f"  Idle Parameters (bypassed branches): {idle_params:,} ({idle_params/1e6:.3f} M, {idle_pct:.1f}%)")
        print(f"  Total Instantiated Parameters:   {stats['total']:,} ({stats['total']/1e6:.3f} M)")
        print("=" * 70)
        return stats

    def encode_gs(self, geo, params):
        """Encodes surface geometry into latent representations."""
        # Prepare positions by scaling
        geo = geo * self.pos_scale_factor
        
        # Sample the initial latent geometry
        latent_geo_pos, idx = sample_geometry(geo, self.num_geo)
        latent_geo_emb = self.pos_encoder(latent_geo_pos)
        
        # Apply encoder blocks
        intermediate_latent_geometries = []
        for block in self.encoder_blocks_gs:
            # Subsample the geometry for geometry cross-attention
            sub_geo_pos, _ = sample_geometry(geo, self.subsampled_geometry_points)
            sub_geo_emb = self.pos_encoder(sub_geo_pos)
            
            # Apply encoder block
            latent_geo_emb, e_ca = block(latent_geo_emb, sub_geo_emb, params, latent_geometry_pos=latent_geo_pos, subsampled_geometry_pos=sub_geo_pos)
            
            # Store for decoder
            intermediate_latent_geometries.append(e_ca)
        
        return intermediate_latent_geometries, latent_geo_pos, idx
    
    def encode_ps(self, geo, latent_geo_pos, idx, physics, params):
        """Encodes surface physics into latent representations."""
        # Sample the initial latent geometry
        latent_physics = physics[:, idx, :]
        
        latent_physics = self.physics_mlp_surface(latent_physics)
        
        latent_geo_emb = self.pos_encoder(latent_geo_pos)
        latent_physics = latent_physics + latent_geo_emb
        
        # Apply encoder blocks
        intermediate_latent_geometries = []
        for block in self.encoder_blocks_ps:
            # Subsample the geometry for geometry cross-attention
            sub_geo_pos, sub_idx = sample_geometry(geo, self.subsampled_geometry_points)
            sub_physics = physics[:, sub_idx, :]
            
            sub_physics = self.physics_mlp_surface(sub_physics)
            
            sub_geo_emb = self.pos_encoder(sub_geo_pos)
            sub_physics = sub_physics + sub_geo_emb
            
            # Apply encoder block
            latent_physics, e_ca = block(latent_physics, sub_physics, params, latent_geometry_pos=latent_geo_pos, subsampled_geometry_pos=sub_geo_pos)
            
            # Store for decoder
            intermediate_latent_geometries.append(e_ca)
        
        return intermediate_latent_geometries
    
    def encode_pv(self, geo, physics, params):
        """Encodes volume physics into latent representations."""
        # Prepare positions by scaling
        geo = geo * self.pos_scale_factor
        
        # Sample the initial latent geometry
        latent_geo_pos, idx = sample_geometry(geo, self.num_geo)
        latent_physics = physics[:, idx, :]
        
        latent_physics = self.physics_mlp_volume(latent_physics)
        
        latent_geo_emb = self.pos_encoder(latent_geo_pos)
        latent_physics = latent_physics + latent_geo_emb
        
        # Apply encoder blocks
        intermediate_latent_geometries = []
        for block in self.encoder_blocks_pv:
            # Subsample the geometry for geometry cross-attention
            sub_geo_pos, sub_idx = sample_geometry(geo, self.subsampled_geometry_points)
            sub_physics = physics[:, sub_idx, :]
            
            sub_physics = self.physics_mlp_volume(sub_physics)
            
            sub_geo_emb = self.pos_encoder(sub_geo_pos)
            sub_physics = sub_physics + sub_geo_emb
            
            # Apply encoder block
            latent_physics, e_ca = block(latent_physics, sub_physics, params, latent_geometry_pos=latent_geo_pos, subsampled_geometry_pos=sub_geo_pos)
            
            # Store for decoder
            intermediate_latent_geometries.append(e_ca)
        
        return intermediate_latent_geometries, latent_geo_pos, idx
    
    def apply_latent_coattention(self, intermediate_s, intermediate_v, pos_s, pos_v):
        """Applies bidirectional cross-modal co-attention between surface and volume latents."""
        if intermediate_s is None or intermediate_v is None or pos_s is None or pos_v is None:
            return intermediate_s, intermediate_v
        new_s, new_v = [], []
        for s_feat, v_feat in zip(intermediate_s, intermediate_v):
            if s_feat is not None and v_feat is not None:
                s_up, v_up = self.latent_coattn(s_feat, v_feat, pos_s, pos_v)
                new_s.append(s_up)
                new_v.append(v_up)
            else:
                new_s.append(s_feat)
                new_v.append(v_feat)
        return new_s, new_v

    def _compute_router_weights(self, query_emb):
        """Computes channel branch routing weights."""
        weights = self.mlp_gate(query_emb)  # (B, Nq, 3 * latent_dim)
        w_gs = weights[:, :, 0 : self.latent_dim]
        w_ps = weights[:, :, self.latent_dim : 2 * self.latent_dim]
        w_pv = weights[:, :, 2 * self.latent_dim : 3 * self.latent_dim]
        return w_gs, w_ps, w_pv

    def decode(self, intermediate_latent_geometries_s, intermediate_latent_physics_s, intermediate_latent_physics_v, latent_geo_pos_s, latent_geo_pos_v, params, query_pos):
        """Decodes latent representations into predictions at query positions."""
        # Prepare positions by scaling
        query_pos = query_pos * self.pos_scale_factor
        query_emb = self.pos_encoder(query_pos)

        has_any_gs = intermediate_latent_geometries_s is not None and any(x is not None for x in intermediate_latent_geometries_s)
        has_any_ps = intermediate_latent_physics_s is not None and any(x is not None for x in intermediate_latent_physics_s)
        has_any_pv = intermediate_latent_physics_v is not None and any(x is not None for x in intermediate_latent_physics_v)

        # Apply decoder blocks
        w_gs, w_ps, w_pv = self._compute_router_weights(query_emb)
        query_embs = []
        if has_any_gs:
            query_emb_1 = query_emb
            for e_gs, block_gs in zip(intermediate_latent_geometries_s, self.decoder_blocks_gs):
                query_emb_1 = block_gs(query_emb_1, e_gs, params, queries_pos=query_pos, latent_geometry_pos=latent_geo_pos_s)
            query_embs.append(w_gs * query_emb_1)
        if has_any_ps:
            query_emb_2 = query_emb
            for e_ps, block_ps in zip(intermediate_latent_physics_s, self.decoder_blocks_ps):
                query_emb_2 = block_ps(query_emb_2, e_ps, params, queries_pos=query_pos, latent_geometry_pos=latent_geo_pos_s)
            query_embs.append(w_ps * query_emb_2)
        if has_any_pv:
            query_emb_3 = query_emb
            for e_pv, block_pv in zip(intermediate_latent_physics_v, self.decoder_blocks_pv):
                query_emb_3 = block_pv(query_emb_3, e_pv, params, queries_pos=query_pos, latent_geometry_pos=latent_geo_pos_v)
            query_embs.append(w_pv * query_emb_3)

        query_emb = torch.sum(torch.stack(query_embs, dim=0), dim=0)

        # Final MLP
        pred = self.mlp_final(query_emb)

        return pred
    
    def forward(self, geo_surface, geo_volume, physics_surface, physics_volume, surf_query_pos, vol_query_pos, params):
        """Forward method for GMPNet model.
        
        Args:
            geo_surface: Input surface geometry with shape (batch size, number points, spatial_dim).
            geo_volume: Input volume geometry with shape (batch size, number points, spatial_dim).
            physics_surface: Input surface physics with shape (batch size, number points, physics_dim).
            physics_volume: Input volume physics with shape (batch size, number points, physics_dim).
            surf_query_pos: Surface query positions with shape (batch size, number surface query points, spatial_dim).
            vol_query_pos: Volume query positions with shape (batch size, number volume query points, spatial_dim).
            params: Conditioning parameters with shape (batch size, cond_dim). If not used, pass None.
        
        Returns:
            tuple: A tuple containing:
                - Surface predictions with shape (batch size, number surface query points, surface_channels).
                - Volume predictions with shape (batch size, number volume query points, volume_channels).
        """
        
        # Encode
        if params is not None and params.ndim == 1:
            params = params.unsqueeze(-1)
        if geo_surface is not None:
            intermediate_latent_geometries_s, latent_geo_pos_s, idx_s = self.encode_gs(geo_surface, params)
            if physics_surface is not None:
                intermediate_latent_physics_s = self.encode_ps(geo_surface, latent_geo_pos_s, idx_s, physics_surface, params)
            else:
                intermediate_latent_physics_s = None
        else:
            intermediate_latent_geometries_s, latent_geo_pos_s, idx_s = None, None, None
            intermediate_latent_physics_s = None
        
        if geo_volume is not None:
            if physics_volume is not None:
                intermediate_latent_physics_v, latent_geo_pos_v, idx_v = self.encode_pv(geo_volume, physics_volume, params)
            else:
                intermediate_latent_physics_v, latent_geo_pos_v, idx_v = None, None, None
        else:
            latent_geo_pos_v, idx_v = None, None
            intermediate_latent_physics_v = None

        # Cross-modal latent co-attention
        if self.use_latent_coattention:
            if intermediate_latent_physics_s is not None and intermediate_latent_physics_v is not None:
                intermediate_latent_physics_s, intermediate_latent_physics_v = self.apply_latent_coattention(
                    intermediate_latent_physics_s, intermediate_latent_physics_v, latent_geo_pos_s, latent_geo_pos_v
                )
            elif intermediate_latent_geometries_s is not None and intermediate_latent_physics_v is not None:
                intermediate_latent_geometries_s, intermediate_latent_physics_v = self.apply_latent_coattention(
                    intermediate_latent_geometries_s, intermediate_latent_physics_v, latent_geo_pos_s, latent_geo_pos_v
                )
                
        # Prepare query positions by concatenating surface and volume query positions
        if surf_query_pos is not None and vol_query_pos is not None:
            query_pos = torch.cat([surf_query_pos, vol_query_pos], dim=1)
        elif surf_query_pos is not None:
            query_pos = surf_query_pos
        elif vol_query_pos is not None:
            query_pos = vol_query_pos
        
        # Decode
        pred = self.decode(intermediate_latent_geometries_s, intermediate_latent_physics_s, intermediate_latent_physics_v, latent_geo_pos_s, latent_geo_pos_v, params, query_pos)
        
        # Split surface and volume predictions
        if surf_query_pos is not None and vol_query_pos is not None:
            pred_surf = pred[:, :surf_query_pos.shape[1], 0:self.surface_channels]
            pred_vol = pred[:, surf_query_pos.shape[1]:, self.surface_channels:]
        elif surf_query_pos is not None and vol_query_pos is None:
            pred_surf = pred[:, :, 0:self.surface_channels] if self.volume_channels > 0 else pred
            pred_vol = torch.zeros(0, 0, 0, device=pred.device, dtype=pred.dtype)
        elif surf_query_pos is None and vol_query_pos is not None:
            pred_surf = torch.zeros(0, 0, 0, device=pred.device, dtype=pred.dtype)
            pred_vol = pred[:, :, self.surface_channels:] if self.surface_channels > 0 else pred
        
        return pred_surf, pred_vol


# Alias for backwards compatibility
SMART = GMPNet