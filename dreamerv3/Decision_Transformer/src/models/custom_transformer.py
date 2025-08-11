# dreamerv3/Decision_Transformer/src/models/custom_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Optional

from jaxtyping import Int

from transformer_lens import HookedTransformer
from transformer_lens.components import Attention, TransformerBlock
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache, HookedTransformerKeyValueCacheEntry
import einops

# --- STEP 1: Attention 레이어가 마스크를 받도록 수정 ---
class MaskableAttention(Attention):
    """
    원본 Attention 클래스를 상속받아, forward 메서드에서 attention_mask를
    처리할 수 있도록 확장합니다.
    """
    # FIX: 부모 클래스의 __init__ 시그니처에 맞게 cfg만 받도록 수정합니다.
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(
        self,
        query_input,
        key_input,
        value_input,
        past_kv_cache_entry: Optional[HookedTransformerKeyValueCacheEntry] = None,
        shortformer_pos_embed: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None, # 인자 추가
    ):
        # 원본 forward 로직의 일부를 가져옵니다.
        q = self.hook_q(
            einops.einsum(
                query_input, self.W_Q,
                "batch pos d_model, n_heads d_model d_head -> batch pos n_heads d_head"
            ) + self.b_Q
        )
        k = self.hook_k(
            einops.einsum(
                key_input, self.W_K,
                "batch pos d_model, n_heads d_model d_head -> batch pos n_heads d_head"
            ) + self.b_K
        )
        v = self.hook_v(
            einops.einsum(
                value_input, self.W_V,
                "batch pos d_model, n_heads d_model d_head -> batch pos n_heads d_head"
            ) + self.b_V
        )

        if past_kv_cache_entry is not None:
            k, v = past_kv_cache_entry.append(k, v)

        attn_scores = (
            einops.einsum(
                q,
                k,
                "batch query_pos head_index d_head, batch key_pos head_index d_head -> batch head_index query_pos key_pos",
            )
            / self.attn_scale
        )

        # --- 핵심 수정 부분 ---
        # 어텐션 마스크를 적용하여 패딩된 부분은 무시하도록 합니다.
        if attention_mask is not None:
            # 마스크 모양을 [batch, seq_len] -> [batch, 1, 1, seq_len]으로 변경
            reshaped_mask = attention_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(reshaped_mask == 0, -1e9)

        pattern = self.hook_pattern(F.softmax(attn_scores, dim=-1))
        z = self.hook_z(torch.einsum("bhqk,bkhv->bqhv", pattern, v))
        
        # 원본 forward 로직의 나머지 부분을 가져옵니다.
        if not self.cfg.use_attn_result:
            out = (
                self.hook_result(
                    torch.einsum("b q h v, h v m -> b q m", z, self.W_O)
                )
                + self.b_O
            )
        else:
            result = self.hook_result(
                torch.einsum("b q h v, h v m -> b q h m", z, self.W_O)
            )
            out = einops.reduce(
                result, "batch position head_index d_model -> batch position d_model", "sum"
            ) + self.b_O
        return out

# --- STEP 2: TransformerBlock이 MaskableAttention을 사용하도록 수정 ---
class MaskableTransformerBlock(TransformerBlock):
    """
    원본 TransformerBlock을 상속받아, MaskableAttention을 사용하고
    forward 메서드에서 attention_mask를 전달하도록 확장합니다.
    """
    def __init__(self, cfg, block_index):
        super().__init__(cfg, block_index)
        # 원본 Attention 대신 우리가 만든 MaskableAttention으로 교체합니다.
        # FIX: MaskableAttention을 초기화할 때 cfg만 전달합니다.
        self.attn = MaskableAttention(cfg)

    def forward(
        self,
        residual: torch.Tensor,
        past_kv_cache_entry: Optional[HookedTransformerKeyValueCacheEntry] = None,
        shortformer_pos_embed: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None, # 인자 추가
    ) -> torch.Tensor:
        # 원본 forward 로직을 가져오되, self.attn 호출 시 attention_mask를 전달합니다.
        attn_out = self.hook_attn_out(
            self.attn(
                self.ln1(residual),
                self.ln1(residual),
                self.ln1(residual),
                past_kv_cache_entry=past_kv_cache_entry,
                shortformer_pos_embed=shortformer_pos_embed,
                attention_mask=attention_mask, # attn에 마스크 전달
            )
        )
        residual = self.hook_resid_mid(residual + attn_out)

        if self.cfg.attn_only:
            return residual
        
        mlp_out = self.hook_mlp_out(self.mlp(self.ln2(residual)))
        residual = self.hook_resid_post(residual + mlp_out)
        return residual

# --- STEP 3: HookedTransformer가 MaskableTransformerBlock을 사용하도록 수정 ---
class MaskableHookedTransformer(HookedTransformer):
    """
    원본 HookedTransformer를 상속받아, MaskableTransformerBlock을 사용하고
    forward 메서드에서 attention_mask를 전달하도록 확장합니다.
    """
    def __init__(self, cfg, tokenizer=None, move_to_device=True):
        # HookedTransformer.__init__이 요구하는 속성들이 cfg에 없을 경우 기본값을 설정해줍니다.
        if not hasattr(cfg, 'n_devices'):
            cfg.n_devices = 1
        if not hasattr(cfg, 'device'):
            cfg.device = None
        if not hasattr(cfg, 'tokenizer_name'):
            cfg.tokenizer_name = None
        if not hasattr(cfg, 'use_hook_tokens'):
            cfg.use_hook_tokens = False
        
        if not hasattr(cfg, 'd_vocab'):
            cfg.d_vocab = -1
        if not hasattr(cfg, 'd_vocab_out'):
            cfg.d_vocab_out = -1

        super().__init__(cfg, tokenizer, move_to_device)
        # 원본 TransformerBlock 대신 우리가 만든 MaskableTransformerBlock으로 교체합니다.
        self.blocks = nn.ModuleList(
            [
                MaskableTransformerBlock(self.cfg, block_index)
                for block_index in range(self.cfg.n_layers)
            ]
        )
        # setup()을 다시 호출하여 새로운 모듈들을 등록합니다.
        self.setup()

    def forward(
        self,
        x: torch.Tensor,
        return_type: Optional[str] = "logits",
        attention_mask: Optional[torch.Tensor] = None,
        past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
        **kwargs,
    ):
        residual = x

        for i, block in enumerate(self.blocks):
            residual = block(
                residual,
                past_kv_cache_entry=past_kv_cache[i] if past_kv_cache is not None else None,
                attention_mask=attention_mask,
            )

        if self.cfg.normalization_type is not None:
            residual = self.ln_final(residual)
        
        if return_type is None:
            return None
        
        return residual
