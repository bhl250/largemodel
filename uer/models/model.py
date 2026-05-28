import torch.nn as nn


class Model(nn.Module):
    """
    Pretraining models consist of three parts:
    - embedding
    - encoder
    - target
    """
    def __init__(self, args, embedding, encoder, target):
        super(Model, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target

        if args.target in ["bert", "mlm"] and args.tie_weights:
            self.target.mlm_linear_2.weight = self.embedding.word_embedding.weight
        elif args.target in ["lm", "t5"] and args.tie_weights:
            self.target.output_layer.weight = self.embedding.word_embedding.weight

        if args.target == "t5" and getattr(args, "share_embedding", False):
            self.target.embedding.word_embedding.weight = self.embedding.word_embedding.weight

    def forward(self, src, tgt, seg):
        emb = self.embedding(src, seg)
        enc_out = self.encoder(emb, seg)
        if isinstance(enc_out, tuple):
            output, aux_loss = enc_out[0], enc_out[1]
        else:
            output, aux_loss = enc_out, None

        loss_info = self.target(output, tgt)
        if aux_loss is not None:
            if isinstance(loss_info, tuple):
                return loss_info + (aux_loss,)
            return loss_info, aux_loss
        return loss_info
