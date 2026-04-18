import os
import time
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import tiktoken

from train_gpt import GPT, GPTConfig
from sft_datasets import SmolTalk, MMLUTask, GSM8KTask, TaskMixture

# torchrun --standalone --nproc_per_node=2 sft.py --checkpoint-filename=model_10699_climbmix_700M.pt
ddp = int(os.environ.get('RANK', -1) != -1)
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

device_type = "cuda" if device.startswith("cuda") else "cpu"
print(f'using device_type: {device_type}')

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

SPECIAL_TOKENS = {
    '<|bos|>': 50256,  # beginning of sequence, same as <|endoftext|>
    '<|user_start|>': 50257,
    '<|user_end|>': 50258,
    '<|assistant_start|>': 50259,
    '<|assistant_end|>': 50260,
}


def render_conversation(conversation, max_tokens):
    ids, mask = [], []

    def add(token_ids, mask_val):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        ids.extend(token_ids)
        mask.extend([mask_val] * len(token_ids))

    messages = conversation['messages']

    # if the first message is system, merge it into the first user message
    if messages[0]['role'] == 'system':
        messages = list(messages)  # make a copy, does not mutate the original conversation dict
        assert messages[1]['role'] == 'user'
        messages[1] = {
            'role': 'user',
            'content': messages[0]['content'] + '\n\n' + messages[1]['content']
        }
        messages = messages[1:]

    # BOS token (not supervised)
    add(SPECIAL_TOKENS['<|bos|>'], 0)

    for i, message in enumerate(messages):
        content = message['content']
        text_ids = enc.encode_ordinary(content)

        if message['role'] == 'user':
            add(SPECIAL_TOKENS['<|user_start|>'], 0)
            add(text_ids, 0)  # user token, don't train on these
            add(SPECIAL_TOKENS['<|user_end|>'], 0)
        elif message['role'] == 'assistant':
            add(SPECIAL_TOKENS['<|assistant_start|>'], 0)
            add(text_ids, 1)  # train on these
            add(SPECIAL_TOKENS['<|assistant_end|>'], 1)

    # Truncate
    ids = ids[:max_tokens]
    mask = mask[:max_tokens]

    return ids, mask


def sft_data_generator(dataset, buffer_size=100):
    dataset_len = len(dataset)
    bos_token = SPECIAL_TOKENS['<|bos|>']
    row_capacity = T + 1
    conv_buffer = []  # list of (token_ids, loss_mask) tuples
    cursor = ddp_rank
    consumed = ddp_rank

    def refill():
        nonlocal cursor
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor % dataset_len]
            ids, mask = render_conversation(conversation, max_tokens=T)
            conv_buffer.append((ids, mask))
            cursor += ddp_world_size  # refill with ddp_world_size steps

    while True:
        rows, mask_rows, row_lengths = [], [], []

        for _ in range(B):
            row, mask_row = [], []
            padded = False

            while len(row) < row_capacity:
                refill()

                remaining = row_capacity - len(row)

                best_idx, best_len = -1, 0

                for i, (conv, _) in enumerate(conv_buffer):
                    if len(conv) <= remaining and len(conv) > best_len:
                        best_idx = i
                        best_len = len(conv)

                if best_idx >= 0:
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                    consumed += ddp_world_size
                else:
                    # Pad the rest (don't crop conversations - we want all tokens)
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    mask_row.extend([0] * remaining)
                    padded = True
                    break

            if padded:
                row_lengths.append(content_len)
            else:
                row_lengths.append(row_capacity)

            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])

        # Build tensors
        batch = torch.tensor(rows, dtype=torch.long)
        inputs = batch[:, :-1].to(device)
        targets = batch[:, 1:].to(device)

        # Apply loss mask: set targets to -1 where mask=0
        mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
        mask_targets = mask_tensor[:, 1:].to(device)
        targets[mask_targets == 0] = -100  # cross_entropy ignores index -100

        yield inputs.to(device), targets.to(device), consumed / dataset_len


# ----- Load pretrained checkpoint -----
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-filename', type=str)
    args = parser.parse_args()
    checkpoint_filename = args.checkpoint_filename

    torch.set_float32_matmul_precision('high')  # use tf32

    checkpoint = torch.load(f'weights/{checkpoint_filename}', weights_only=False, map_location=device)
    config = checkpoint['config']
    # todo we saved the compiled model previously, we no longer do that
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model'].items()}
    model = GPT(config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model
    if ddp_rank == 0:
        print(
            f"Loaded pretrained model {checkpoint_filename} (step: {checkpoint['step']}, val_loss:{checkpoint['val_loss']:.4f})")

    enc = tiktoken.get_encoding('gpt2')

    # --- Hyperparameters ---
    B = 32
    T = 1024
    total_batch_size = 524288
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    learning_rate = 6e-5  # 10x lower than pretraining max_lr
    weight_decay = 0.0

    train_dataset = TaskMixture([
        SmolTalk(split='train'),  # 460K general conversation
        *[MMLUTask(subset="all", split='auxiliary_train') for _ in range(3)],  # 100K x 3
        *[GSM8KTask(subset="main", split='train') for _ in range(4)],  # 8K x 4
    ])
    val_dataset = TaskMixture([
        SmolTalk(split="test"),
        MMLUTask(subset="all", split="test"),
        GSM8KTask(subset="main", split="test"),
    ])

    dataset_size = len(train_dataset)
    if ddp_rank == 0:
        print(f'SFT training mixture: {dataset_size:,} conversations')

    train_loader = sft_data_generator(train_dataset)
    optimizer = raw_model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate,
                                               device_type=device_type, betas=(0.9, 0.95), eps=1e-8)


    # Learning rate schedule (linear warmup, constant, linear warmdown)
    # Same shape as base_train but uses progress (0→1) instead of absolute step counts,
    # because SFT doesn't always know num_iterations in advance (dataset-driven stopping).
    def get_lr_multiplier(progress, warmup_ratio=0.0, warmdown_ratio=0.5, final_lr_frac=0.0):
        if progress < warmup_ratio:
            return (progress + 1e-8) / warmup_ratio
        elif progress <= 1.0 - warmdown_ratio:
            return 1.0
        else:
            decay = (progress - (1.0 - warmdown_ratio)) / warmdown_ratio
            return (1 - decay) * 1.0 + decay * final_lr_frac


    def eval_and_save(step):
        model.eval()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loader = sft_data_generator(val_dataset)

            for micro_step in range(grad_accum_steps):
                x, y, _ = next(val_loader)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / grad_accum_steps
                val_loss_accum += loss.detach()

            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

            if ddp_rank == 0:
                print(f'sft val loss: {val_loss_accum.item():.4f}')

                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, f'weights/sft_{checkpoint_filename}')


    step = 0
    x, y, progress = next(train_loader)
    while True:
        t0 = time.time()

        model.train()
        optimizer.zero_grad()

        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss.backward()
            loss_accum += loss.detach()
            x, y, progress = next(train_loader)  # progress can be greater than 1.0

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Step the optimizer
        lr = learning_rate * get_lr_multiplier(progress)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        step += 1

        if device_type == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        dt = t1 - t0

        if ddp_rank == 0:
            print(
                f'sft step {step}, loss: {loss_accum:.6f}, lr: {lr:.4e}, progress: {progress * 100:.1f}%, time: {dt * 1000:.2f}ms')

        if step % 5000 == 0:
            eval_and_save(step)

        if progress > 1:
            break

    eval_and_save(step)

    if ddp:
        destroy_process_group()
