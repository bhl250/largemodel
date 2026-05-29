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

For BERT pretraining, the main TensorBoard view has 22 primary items to watch:
1 computation graph, 7 core scalar curves, and 14 grouped norm curves.

| Count | TensorBoard item | Meaning |
| --- | --- | --- |
| 1 | `Graphs` | The traced model computation graph. Use this to confirm the forward path contains embedding, Transformer/MoE blocks, and the pretraining target. |
| 1 | `loss/total` | Total optimized loss: `loss/mlm + loss/nsp + loss/moe_aux`. This is the first curve to watch for overall convergence. |
| 1 | `loss/mlm` | Masked language modeling loss. This should trend downward during useful pretraining. |
| 1 | `loss/nsp` | Sentence-pair / next-segment prediction loss used by the BERT target in this UER code. |
| 1 | `loss/moe_aux` | MoE routing auxiliary loss from load balancing and router z-loss. This should stay finite and not dominate the total loss. |
| 1 | `accuracy/mlm` | MLM token prediction accuracy on masked positions. It is usually low early in training and should improve gradually. |
| 1 | `accuracy/nsp` | NSP / sentence-pair classification accuracy. |
| 1 | `optim/lr` | Learning-rate schedule. Use this to check warmup and decay behavior. |
| 7 | `param_group_norm/*` | Parameter norms for major model groups: `embedding`, `attention`, `moe_router`, `moe_experts`, `moe_ff_after`, `layer_norm`, `target_head`. Sudden spikes can indicate unstable weights. |
| 7 | `grad_group_norm/*` | Gradient norms for the same 7 groups. Sudden spikes or all-zero gradients are the main warning signs. |

In addition to those 22 primary items, the trainer also writes detailed
per-parameter and histogram views:

| TensorBoard item | Write frequency | Meaning |
| --- | --- | --- |
| `param_norm/<parameter_name>` | every 100 steps | L2 norm of each individual parameter tensor. Use this when a grouped norm looks abnormal and you need to locate the exact layer. |
| `grad_norm/<parameter_name>` | every 100 steps | L2 norm of each individual gradient tensor. Use this to inspect whether router/expert/attention gradients are flowing. |
| `hist_params/*` | every 1000 steps | Distribution histogram for selected important parameters: embeddings, attention output projection, MoE router, first expert, and target head. |
| `hist_grads/*` | every 1000 steps | Distribution histogram for selected gradients. Watch for extreme outliers, all-zero distributions, or exploding spread. |
| `Text/run/*` | once at startup | Output path, training state path, and MoE hyperparameters. |
| `Text/graph/status` | once at graph trace | Whether TensorBoard graph tracing succeeded. If graph tracing fails, training still continues. |

If TensorBoard graph tracing fails for any reason, training continues and the
failure reason is written as TensorBoard text under:

```text
graph/status
```

## 7. One-Command Training

Before starting the formal run, archive the previous 1000-step trial run:

```bash
python save_previous_run.py
```

This moves the old trial artifacts into:

```text
saved_runs/moe_pretrain_<timestamp>/
```

It saves the old model checkpoints, resumable state, PID file, TensorBoard logs,
and background log. It also clears the active `models/moe_pretrain_latest_state.pt`
and `runs/moe_pretrain/` locations so the formal run starts from a fresh state
instead of resuming the 1000-step trial.

Use:

```bash
python exutive.py
```

This command first tries to create or update a systemd service named:

```text
etbert-moe-pretrain.service
```

If systemd is available, it starts the service and returns immediately. The
actual training process runs under systemd, not inside your SSH session.

Some school server environments are containers and do not boot with systemd as
PID 1. In that case `systemctl` prints:

```text
System has not been booted with systemd as init system (PID 1). Can't operate.
```

`exutive.py` handles this automatically. It falls back to a detached background
process using `start_new_session=True`, writes the process id to
`models/moe_pretrain.pid`, and writes logs to
`runs/moe_pretrain/background_train.log`. You can disconnect from the remote
server after either startup mode has succeeded.

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

If the script falls back to detached-process mode, use:

```bash
cat models/moe_pretrain.pid
ps -p "$(cat models/moe_pretrain.pid)"
tail -f runs/moe_pretrain/background_train.log
kill "$(cat models/moe_pretrain.pid)"
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
steps = 100000
batch_size = 8
seq_length = 128
moe_experts = 4
moe_top_k = 2
learning_rate = 2e-5
report_steps = 50
state_save_steps = 100
save_checkpoint_steps = 5000
tensorboard_param_steps = 100
tensorboard_histogram_steps = 1000
```

Outputs:

```text
models/moe_pre-trained_model.bin-5000
models/moe_pre-trained_model.bin-10000
...
models/moe_pre-trained_model.bin-100000
models/moe_pretrain_latest_state.pt
models/moe_pretrain.pid
runs/moe_pretrain/
```

The final model checkpoint for this formal 100000-step run is:

```text
models/moe_pre-trained_model.bin-100000
```

The resumable training state is:

```text
models/moe_pretrain_latest_state.pt
```

TensorBoard event files are written under:

```text
runs/moe_pretrain/
```

If systemd is unavailable and detached-process mode is used, the main training
stdout/stderr log is:

```text
runs/moe_pretrain/background_train.log
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

## 10. Fine-Tuning

After MoE pretraining finishes, CSTNET packet-level fine-tuning uses:

```text
datasets/cstnet-tls1.3/packet/train_dataset.tsv
datasets/cstnet-tls1.3/packet/valid_dataset.tsv
datasets/cstnet-tls1.3/packet/test_dataset.tsv
```

These files must contain `label` and `text_a` columns. The downloaded CSTNET
packet dataset has 120 labels, with labels from 0 to 119.

Start fine-tuning in the background:

```bash
python finetune_exutive.py
```

The runner uses:

```text
pretrained_model = models/moe_pre-trained_model.bin-100000
output_model = models/moe_finetuned_cstnet_packet.bin
epochs_num = 5
batch_size = 32
seq_length = 128
learning_rate = 2e-5
moe_experts = 4
moe_top_k = 2
moe_aux_weight = 0.01
```

The fine-tuning process is detached from the SSH session, like pretraining. Its
PID and logs are:

```text
models/moe_finetune.pid
runs/moe_finetune_cstnet_packet/background_finetune.log
```

Check status:

```bash
ps -p "$(cat models/moe_finetune.pid)"
```

Follow logs:

```bash
tail -f runs/moe_finetune_cstnet_packet/background_finetune.log
```

Stop fine-tuning:

```bash
kill "$(cat models/moe_finetune.pid)"
```

Fine-tuning TensorBoard events are written under:

```text
runs/moe_finetune_cstnet_packet/
```

The TensorBoard scalar curves include:

```text
train/loss
train/lr
dev/loss
dev/acc
dev/macro_precision
dev/macro_recall
dev/macro_f1
test/loss
test/acc
test/macro_precision
test/macro_recall
test/macro_f1
```

The fine-tuning script saves the best model by dev accuracy to:

```text
models/moe_finetuned_cstnet_packet.bin
```

For inference after fine-tuning:

```bash
python inference/run_classifier_infer.py \
  --load_model_path models/moe_finetuned_cstnet_packet.bin \
  --vocab_path models/encryptd_vocab.txt \
  --test_path datasets/cstnet-tls1.3/packet/nolabel_test_dataset.tsv \
  --prediction_path datasets/cstnet-tls1.3/packet/moe_prediction.tsv \
  --labels_num 120 \
  --embedding word_pos_seg \
  --encoder transformer \
  --mask fully_visible \
  --seq_length 128 \
  --moe_experts 4 \
  --moe_top_k 2
```

## 11. Important Notes

The one-command runner reads the already-preprocessed file:

```text
dataset.pt
```

It does not regenerate `dataset.pt` from raw traffic or corpus data.

The project currently starts from scratch for MoE pretraining. The previous
server checkpoint `models/pre-trained_model.bin-10` was identified as an old
MoE checkpoint, not a normal FFN ET-BERT checkpoint, and should not be used as
the clean starting point for this run.
