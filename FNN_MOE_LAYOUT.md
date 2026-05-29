# FNN / MoE Baseline Layout

This repository now keeps the MoE code path and the official FNN baseline code
path side by side.

## Directory Layout

```text
MOE/
FNN/
dataset.pt
models/
datasets/
runs/
FNNexutive.py
FNNfinetune.py
exutive.py
finetune_exutive.py
```

`MOE/` is a snapshot of the MoE version that was used for the completed MoE
pretraining and fine-tuning experiments.

`FNN/` is based on the official ET-BERT code from:

```text
https://github.com/linwhitehat/ET-BERT
```

The core FNN difference is in:

```text
FNN/uer/layers/transformer.py
```

It uses the ordinary Transformer feed-forward block:

```text
PositionwiseFeedForward
```

The MoE path remains in:

```text
MOE/uer/layers/transformer.py
```

It uses:

```text
SparseMoEBlock(MoE(...))
```

## Shared Server Data

Large data and generated artifacts are shared from the repository root. They
are not duplicated into `MOE/` or `FNN/`.

Pretraining input:

```text
dataset.pt
models/encryptd_vocab.txt
models/bert/base_config.json
```

Fine-tuning input:

```text
datasets/cstnet-tls1.3/packet/train_dataset.tsv
datasets/cstnet-tls1.3/packet/valid_dataset.tsv
datasets/cstnet-tls1.3/packet/test_dataset.tsv
datasets/cstnet-tls1.3/packet/nolabel_test_dataset.tsv
```

## FNN Pretraining

Start the FNN baseline pretraining:

```bash
python FNNexutive.py
```

The script runs in a detached background process with `start_new_session=True`,
so disconnecting from SSH does not stop the job.

Default FNN pretraining settings:

```text
total_steps = 100000
batch_size = 8
seq_length = 128
learning_rate = 2e-5
save_checkpoint_steps = 5000
state_save_steps = 100
report_steps = 50
```

Output:

```text
models/FNN_pre-trained_model.bin-5000
models/FNN_pre-trained_model.bin-10000
...
models/FNN_pre-trained_model.bin-100000
models/FNN_pretrain_latest_state.pt
runs/FNN_pretrain/
```

Monitor:

```bash
ps -p "$(cat models/FNN_pretrain.pid)"
tail -f runs/FNN_pretrain/background_train.log
```

## FNN Fine-Tuning

After FNN pretraining finishes, start FNN fine-tuning:

```bash
python FNNfinetune.py
```

`FNNfinetune.py` runs fine-tuning. It uses:

```text
models/FNN_pre-trained_model.bin-100000
datasets/cstnet-tls1.3/packet/train_dataset.tsv
datasets/cstnet-tls1.3/packet/valid_dataset.tsv
datasets/cstnet-tls1.3/packet/test_dataset.tsv
```

Default FNN fine-tuning settings:

```text
epochs_num = 5
batch_size = 32
seq_length = 128
learning_rate = 2e-5
```

Output:

```text
models/FNN_finetuned_cstnet_packet.bin
runs/FNN_finetune_cstnet_packet/
```

Monitor:

```bash
ps -p "$(cat models/FNN_finetune.pid)"
tail -f runs/FNN_finetune_cstnet_packet/background_finetune.log
```

## Existing MoE Commands

The root MoE runners are unchanged:

```bash
python exutive.py
python finetune_exutive.py
```

Those scripts continue to use the root MoE code path and output:

```text
models/moe_pre-trained_model.bin-100000
models/moe_finetuned_cstnet_packet.bin
runs/moe_pretrain/
runs/moe_finetune_cstnet_packet/
```

## TensorBoard

Both FNN and MoE runners use:

```text
http://127.0.0.1:6007
```

They all launch TensorBoard with:

```text
--logdir runs
--host 0.0.0.0
--port 6007
```

Expected run directories:

```text
runs/moe_pretrain/
runs/moe_finetune_cstnet_packet/
runs/FNN_pretrain/
runs/FNN_finetune_cstnet_packet/
```
