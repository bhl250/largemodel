# ET-BERT MoE Training Notes

This document records the local changes made to convert the ET-BERT/UER
Transformer feed-forward path into a MoE path, make pretraining runnable again,
add TensorBoard monitoring, and provide a one-command server runner.

## 1. What Changed

The original ET-BERT codebase used the UER-style Transformer encoder:

```text
Embedding -> TransformerEncoder -> Target
```

Each Transformer layer used one ordinary feed-forward network after attention.
The modified version replaces that feed-forward block with a sparse MoE block
from `st_moe_pytorch`.

The main changed files are:

```text
uer/layers/transformer.py
uer/encoders/transformer_encoder.py
uer/decoders/transformer_decoder.py
uer/models/model.py
uer/models/__init__.py
uer/trainer.py
uer/opts.py
pre-training/pretrain.py
exutive.py
requirements.txt
```

`uer/models/model.py` and `uer/models/__init__.py` were added because the
pretraining path imports `uer.models.model.Model`, but the directory was missing
from the repository.

## 2. Before: Ordinary FFN Path

Before the MoE change, each Transformer layer had one feed-forward module.
The layer-level data flow was:

```text
hidden
  -> self attention
  -> residual + layer norm
  -> ordinary FFN
  -> residual + layer norm
  -> output
```

For post-layernorm mode, the conceptual computation was:

```text
A = SelfAttention(H)
U = LayerNorm(H + Dropout(A))
F = FFN(U)
O = LayerNorm(U + Dropout(F))
```

The ordinary FFN was:

```text
FFN(x) = W2 * act(W1 * x + b1) + b2
```

For ET-BERT base-style dimensions:

```text
hidden_size = 768
feedforward_size = 3072
```

So the ordinary FFN was roughly:

```text
W1: 768  -> 3072
W2: 3072 -> 768
```

The model returned only the task outputs:

```text
TransformerEncoder(emb, seg) -> hidden
Target(hidden, tgt) -> task loss and metrics
```

For BERT pretraining, the target returned:

```text
(loss_mlm, loss_nsp, correct_mlm, correct_nsp, denominator)
```

The trainer optimized:

```text
loss_total = loss_mlm + loss_nsp
```

## 3. After: MoE FFN Path

The feed-forward block is now:

```python
SparseMoEBlock(
    moe=MoE(
        dim=args.hidden_size,
        num_experts=args.moe_experts,
        expert_hidden_mult=args.feedforward_size // args.hidden_size,
        gating_top_n=args.moe_top_k,
        balance_loss_coef=args.moe_balance_coef,
        router_z_loss_coef=args.moe_z_loss_coef
    )
)
```

The default MoE settings are:

```text
moe_experts = 4
moe_top_k = 2
moe_balance_coef = 0.01
moe_z_loss_coef = 0.001
```

`st_moe_pytorch` requires `moe_top_k >= 2`, so `moe_top_k=1` is invalid for this
implementation.

The new layer-level data flow is:

```text
hidden
  -> self attention
  -> residual + layer norm
  -> sparse MoE FFN
  -> residual + layer norm
  -> output, moe_aux_loss
```

For post-layernorm mode:

```text
A = SelfAttention(H)
U = LayerNorm(H + Dropout(A))
M, L_aux = MoE(U)
O = LayerNorm(U + Dropout(M))
```

The MoE module can be thought of as:

```text
g(x) = Router(x)
TopK(x) = top-k experts selected by g(x)

MoE(x) = sum_{e in TopK(x)} p_e(x) * Expert_e(x)
```

Each expert is an independent feed-forward network. The experts are not just
aliases of one shared FFN. If training starts from scratch, they are newly
initialized parameters.

The auxiliary loss is provided by `st_moe_pytorch`:

```text
L_aux = L_balance + L_router_z
```

The coefficients are controlled by:

```text
moe_balance_coef
moe_z_loss_coef
```

## 4. Aux Loss Propagation

Before this fix, `TransformerLayer` internally computed MoE auxiliary loss, but
the value did not propagate back to the trainer. That meant the training loss did
not actually include MoE routing regularization.

The fixed propagation is:

```text
TransformerLayer
  -> returns (output, layer_aux_loss)

TransformerEncoder
  -> sums layer_aux_loss across all layers
  -> returns (hidden, total_aux_loss)

Model
  -> calls target(hidden, tgt)
  -> appends total_aux_loss to the target outputs

Trainer
  -> separates target outputs from aux loss
  -> optimizes task loss + aux loss
```

For BERT pretraining after the change:

```text
Target(hidden, tgt)
  -> (loss_mlm, loss_nsp, correct_mlm, correct_nsp, denominator)

Model(...)
  -> (loss_mlm, loss_nsp, correct_mlm, correct_nsp, denominator, loss_moe_aux)
```

The trainer now optimizes:

```text
loss_total = loss_mlm + loss_nsp + loss_moe_aux
```

For classification fine-tuning, the classifier uses:

```text
loss_total = loss_cls + moe_aux_weight * loss_moe_aux
```

The current classifier default is:

```text
moe_aux_weight = 0.01
```

## 5. Mathematical Comparison

### Original Transformer FFN

For token representation `x`:

```text
FFN(x) = W2 * sigma(W1 * x + b1) + b2
```

Layer output:

```text
H' = LN(H + Attn(H))
O  = LN(H' + FFN(H'))
```

Training objective for BERT pretraining:

```text
L = L_MLM + L_NSP
```

### MoE Transformer FFN

Router scores:

```text
s(x) = W_r * x
```

Top-k selected experts:

```text
E_k(x) = TopK(s(x), k)
```

Router probabilities over selected experts:

```text
p_e(x) = softmax(s_e(x)), e in E_k(x)
```

MoE output:

```text
MoE(x) = sum_{e in E_k(x)} p_e(x) * Expert_e(x)
```

Layer output:

```text
H' = LN(H + Attn(H))
O  = LN(H' + MoE(H'))
```

Training objective:

```text
L = L_MLM + L_NSP + L_aux
```

where:

```text
L_aux = balance_coef * L_balance + z_loss_coef * L_router_z
```

The exact internal definitions of `L_balance` and `L_router_z` are implemented
inside `st_moe_pytorch`.

## 6. TensorBoard Monitoring

TensorBoard logging is now required for pretraining. The default log directory is:

```text
runs/moe_pretrain
```

The one-command runner starts TensorBoard on:

```text
0.0.0.0:6007
```

The logged information includes:

```text
Graph:
  - TensorBoard computation graph

Loss:
  - loss/total
  - loss/mlm
  - loss/nsp
  - loss/moe_aux

Accuracy:
  - accuracy/mlm
  - accuracy/nsp

Optimizer:
  - optim/lr

Parameter norms:
  - param_norm/<parameter_name>
  - param_group_norm/embedding
  - param_group_norm/attention
  - param_group_norm/moe_router
  - param_group_norm/moe_experts
  - param_group_norm/moe_ff_after
  - param_group_norm/layer_norm
  - param_group_norm/target_head

Gradient norms:
  - grad_norm/<parameter_name>
  - grad_group_norm/embedding
  - grad_group_norm/attention
  - grad_group_norm/moe_router
  - grad_group_norm/moe_experts
  - grad_group_norm/moe_ff_after
  - grad_group_norm/layer_norm
  - grad_group_norm/target_head

Histograms:
  - selected parameter distributions
  - selected gradient distributions
```

If TensorBoard graph tracing fails for any reason, training continues and the
failure reason is written as TensorBoard text under:

```text
graph/status
```

## 7. One-Command Training

Use:

```bash
python exutive.py
```

This command creates or updates a systemd service named:

```text
etbert-moe-pretrain.service
```

Then it starts the service and returns immediately. The actual training process
runs under systemd, not inside your SSH session. You can disconnect from the
remote server after the service has started.

The script assumes the server project has:

```text
dataset.pt
models/encryptd_vocab.txt
models/bert/base_config.json
```

It uses:

```text
Physical GPU: 1
Training process GPU rank: 0
TensorBoard port: 6007
```

`exutive.py` sets `CUDA_VISIBLE_DEVICES=1`, so the training process only sees
physical GPU 1. Inside that process, the visible GPU is addressed as
`--gpu_ranks 0`.

Internally, systemd runs:

```bash
python exutive.py --run-training
```

Do not pass `--run-training` yourself unless you intentionally want to run the
training process in the current terminal.

Check training status:

```bash
systemctl status etbert-moe-pretrain.service
```

Follow logs:

```bash
journalctl -u etbert-moe-pretrain.service -f
```

Stop training:

```bash
systemctl stop etbert-moe-pretrain.service
```

If the script is run by a non-root user, it uses a user-level service instead.
In that case the commands become:

```bash
systemctl --user status etbert-moe-pretrain.service
journalctl --user -u etbert-moe-pretrain.service -f
systemctl --user stop etbert-moe-pretrain.service
```

On your school server examples you were running as `root`, so the normal
system-level commands above should be the expected path.

### Stage 1: Smoke Test

The script first runs a smoke test:

```text
steps = 10
batch_size = 1
output = models/moe_smoke_test.bin
tensorboard = runs/moe_smoke_test
```

If the smoke test succeeds, the temporary smoke checkpoint and state file are
removed.

### Stage 2: Main Training

Then it starts main training:

```text
steps = 1000
batch_size = 8
seq_length = 128
moe_experts = 4
moe_top_k = 2
learning_rate = 2e-5
report_steps = 10
state_save_steps = 50
save_checkpoint_steps = 1000
```

Outputs:

```text
models/moe_pre-trained_model.bin-1000
models/moe_pretrain_latest_state.pt
runs/moe_pretrain/
```

The final model checkpoint for this 1000-step run is:

```text
models/moe_pre-trained_model.bin-1000
```

The resumable training state is:

```text
models/moe_pretrain_latest_state.pt
```

TensorBoard event files are written under:

```text
runs/moe_pretrain/
```

## 8. Resume Behavior

The runner saves a resumable training state:

```text
models/moe_pretrain_latest_state.pt
```

This state contains:

```text
model state_dict
optimizer state_dict
scheduler state_dict
current step
```

If training is interrupted, run the same command again:

```bash
python exutive.py
```

If `models/moe_pretrain_latest_state.pt` exists, the script skips the smoke test
and resumes the main training automatically.

The trainer also writes this state on normal exit and in the `finally` path when
training is interrupted or errors out. The saved step is the last fully completed
training step, so rerunning `python exutive.py` continues from the next step.

## 9. Viewing TensorBoard

The script starts TensorBoard on port `6007`.

If your local machine maps server port 6007, open:

```text
http://127.0.0.1:6007
```

If using SSH port forwarding manually:

```bash
ssh -L 6007:127.0.0.1:6007 user@server
```

Then open:

```text
http://127.0.0.1:6007
```

## 10. Important Notes

The one-command runner reads the already-preprocessed file:

```text
dataset.pt
```

It does not regenerate `dataset.pt` from raw traffic or corpus data.

The project currently starts from scratch for MoE pretraining. The previous
server checkpoint `models/pre-trained_model.bin-10` was identified as an old
MoE checkpoint, not a normal FFN ET-BERT checkpoint, and should not be used as
the clean starting point for this run.
