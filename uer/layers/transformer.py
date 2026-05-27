import torch.nn as nn
from uer.layers.layer_norm import LayerNorm, T5LayerNorm
from uer.layers.position_ffn import PositionwiseFeedForward, GatedFeedForward
from uer.layers.multi_headed_attn import MultiHeadedAttention
from uer.layers.relative_position_embedding import RelativePositionEmbedding
from st_moe_pytorch import MoE, SparseMoEBlock


class TransformerLayer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.layernorm_positioning = args.layernorm_positioning

        attention_head_size = getattr(
            args, "attention_head_size", args.hidden_size // args.heads_num
        )

        has_bias = bool(1 - args.remove_transformer_bias)
        with_scale = bool(1 - args.remove_attention_scale)

        self.self_attn = MultiHeadedAttention(
            args.hidden_size,
            args.heads_num,
            attention_head_size,
            args.dropout,
            has_bias=has_bias,
            with_scale=with_scale
        )
        self.dropout_1 = nn.Dropout(args.dropout)

        # === MoE Feed Forward ===
        self.feed_forward = SparseMoEBlock(
            moe=MoE(
                dim=args.hidden_size,
                num_experts=args.moe_experts,
                expert_hidden_mult=args.feedforward_size // args.hidden_size,
                gating_top_n=args.moe_top_k,
                balance_loss_coef=args.moe_balance_coef,
                router_z_loss_coef=args.moe_z_loss_coef
            )
        )
        self.dropout_2 = nn.Dropout(args.dropout)

        if args.layernorm == "t5":
            self.layer_norm_1 = T5LayerNorm(args.hidden_size)
            self.layer_norm_2 = T5LayerNorm(args.hidden_size)
        else:
            self.layer_norm_1 = LayerNorm(args.hidden_size)
            self.layer_norm_2 = LayerNorm(args.hidden_size)

        self.moe_aux_loss = 0.0

    def forward(self, hidden, mask, position_bias=None):

        if self.layernorm_positioning == "post":
            inter = self.dropout_1(
                self.self_attn(hidden, hidden, hidden, mask, position_bias)
            )
            inter = self.layer_norm_1(inter + hidden)

            moe_ret = self.feed_forward(inter)
            self.moe_aux_loss += moe_ret.total_aux_loss

            output = self.dropout_2(moe_ret.outputs)
            output = self.layer_norm_2(output + inter)
        else:
            normed = self.layer_norm_1(hidden)
            inter = self.dropout_1(
                self.self_attn(normed, normed, normed, mask, position_bias)
            )
            hidden = hidden + inter

            normed = self.layer_norm_2(hidden)
            moe_ret = self.feed_forward(normed)
            self.moe_aux_loss += moe_ret.total_aux_loss

            output = self.dropout_2(moe_ret.outputs) + hidden

        return output

class TransformerDecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.layernorm_positioning = args.layernorm_positioning

        attention_head_size = getattr(
            args, "attention_head_size", args.hidden_size // args.heads_num
        )

        has_bias = bool(1 - args.remove_transformer_bias)
        with_scale = bool(1 - args.remove_attention_scale)

        self.self_attn = MultiHeadedAttention(
            args.hidden_size,
            args.heads_num,
            attention_head_size,
            args.dropout,
            has_bias=has_bias,
            with_scale=with_scale
        )
        self.dropout_1 = nn.Dropout(args.dropout)

        self.context_attn = MultiHeadedAttention(
            args.hidden_size,
            args.heads_num,
            attention_head_size,
            args.dropout,
            has_bias=has_bias,
            with_scale=with_scale
        )
        self.dropout_2 = nn.Dropout(args.dropout)

        # === MoE FFN ===
        self.feed_forward = SparseMoEBlock(
            moe=MoE(
                dim=args.hidden_size,
                num_experts=args.moe_experts,
                expert_hidden_mult=args.feedforward_size // args.hidden_size,
                gating_top_n=args.moe_top_k,
                balance_loss_coef=args.moe_balance_coef,
                router_z_loss_coef=args.moe_z_loss_coef
            )
        )
        self.dropout_3 = nn.Dropout(args.dropout)

        if args.layernorm == "t5":
            self.layer_norm_1 = T5LayerNorm(args.hidden_size)
            self.layer_norm_2 = T5LayerNorm(args.hidden_size)
            self.layer_norm_3 = T5LayerNorm(args.hidden_size)
        else:
            self.layer_norm_1 = LayerNorm(args.hidden_size)
            self.layer_norm_2 = LayerNorm(args.hidden_size)
            self.layer_norm_3 = LayerNorm(args.hidden_size)

        self.moe_aux_loss = 0.0

    def forward(
        self,
        hidden,
        encoder_hidden,
        mask_decoder,
        mask_encoder,
        self_position_bias=None,
        context_position_bias=None
    ):
        if self.layernorm_positioning == "post":
            query = self.dropout_1(
                self.self_attn(hidden, hidden, hidden, mask_decoder, self_position_bias)
            )
            query = self.layer_norm_1(query + hidden)

            mid = self.dropout_2(
                self.context_attn(
                    encoder_hidden, encoder_hidden, query, mask_encoder, context_position_bias
                )
            )
            mid = self.layer_norm_2(mid + query)

            moe_ret = self.feed_forward(mid)
            self.moe_aux_loss += moe_ret.total_aux_loss

            output = self.dropout_3(moe_ret.outputs)
            output = self.layer_norm_3(output + mid)
        else:
            normed = self.layer_norm_1(hidden)
            query = self.dropout_1(
                self.self_attn(normed, normed, normed, mask_decoder, self_position_bias)
            )
            hidden = hidden + query

            normed = self.layer_norm_2(hidden)
            mid = self.dropout_2(
                self.context_attn(
                    encoder_hidden, encoder_hidden, normed, mask_encoder, context_position_bias
                )
            )
            hidden = hidden + mid

            normed = self.layer_norm_3(hidden)
            moe_ret = self.feed_forward(normed)
            self.moe_aux_loss += moe_ret.total_aux_loss

            output = self.dropout_3(moe_ret.outputs) + hidden

        return output
