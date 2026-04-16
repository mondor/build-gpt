import os
import math
import inspect
import time
from dataclasses import dataclass
from dataloader import DataLoaderLite

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import tiktoken

from hellaswag import render_example, iterate_examples, get_most_likely_row

# torchrun --standalone --nproc_per_node=8 train_gpt2.py
# set up distributed data parallel
# torchrun command sets the env variables
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
    device = 'cuda' if torch.cuda.is_available() else (
        'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')

device_type = "cuda" if device.startswith("cuda") else "cpu"
print(f'using device_type: {device_type}')

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


# -------------------GPT MODEL------------------
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257  # 50,000 BPE merges + 256 bytes tokens + 1 + 1 <|endoftext|> token
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.GPT_SCALE_INIT = 1  # flag to apply initialization

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)  # 3 because we have 3 weights q,k,v
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.GPT_SCALE_INIT = 1  # flag to apply initialization
        # not really a bias, more of a mask. follow the OpenAI naming to test our model by loading against their weight.
        self.register_buffer('bias',
                             torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size,
                                                                                               config.block_size))

    def forward(self, x):
        B, T, C = x.shape  # (B, T, C) - batch, time_step, channel(n_embed)
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embed, dim=2)  # (B, T, C) - for q,k,v

        hs = C // self.n_head  # head size
        nh = self.n_head  # num heads

        q = q.view(B, T, nh, hs)  # (B, T, nh, hs)
        q = q.transpose(1, 2)  # (B, nh, T, hs) - turning n_head into batch
        k = k.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)

        """
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        """
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # Flash attention, kernel fusion.

        y = y.transpose(1, 2).contiguous()  # (B, T, nh, hs)
        y = y.view(B, T, C)

        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embed),
            wpe=nn.Embedding(config.block_size, config.n_embed),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embed)
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'GPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5  # 2 residual per layer
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot foward sequence of {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)

        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)  # final layer norm
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print('loading weights from pretrained gpt %s' % model_type)

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embed=768),  # 124M
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embed=1024),  # 350M
            'gpt2-large': dict(n_layer=36, n_head=20, n_embed=1280),  # 774M
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embed=1600),  # 1558M
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard the mask (bias)

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape  # .shape[::-1] reverse the shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape, f"{k}: {sd_hf[k].shape} != {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that required grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f'num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters')
        print(f'num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters')

        # create adamW optimiser and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f'using fused adamW: {use_fused}')
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        return optimizer


# ------------------------TRAIN---------------------------
if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')  # use tf32

    total_batch_size = 524288  # 2**19, ~0.5M tokens in the original gpt2 paper
    B = 64  # micro batch size
    T = 1024
    assert total_batch_size % (B * T * ddp_world_size) == 0
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if ddp_rank == 0:
        print(
            f"total desired batch size: {total_batch_size}, calculated gradient accumulation steps: {grad_accum_steps}")

    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    use_compile = False
    if use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])  # averages and deposit the gradient on each GPU
    raw_model = model.module if ddp else model

    enc = tiktoken.get_encoding("gpt2")

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715  # gpt paper says warmup 375e6 tokens, 375e6/2**19 = 715
    max_steps = 19073  # we want to train 10B tokens total, 10e9/2**19 = 19073


    def get_lr(it):
        # linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        # if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)


    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

    log_file = f"log.txt"
    with open(log_file, 'w') as f:  # open for writing to clear the file
        pass

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # val
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
                if ddp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
                if ddp_rank == 0:
                    print(f'val loss: {val_loss_accum.item():.4f}')
                    with open(log_file, 'a') as f:
                        f.write(f'{step} val {val_loss_accum.item():.4f}\n')
                    if step > 0 and (step % 5000 == 0 or last_step):
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'config': raw_model.config,
                            'step': step,
                            'val_loss': val_loss_accum.item()
                        }
                        # you might also want to add optimizer.state_dict() and
                        # rng seeds etc., if you wanted to more exactly resume training
                        torch.save(checkpoint, f'model_{step:05d}.pt')

        # hellaswag val
        if (step % 250 == 0 or last_step) and (not use_compile):
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples('val')):
                # only process example where i % ddp_world_size == ddp_rank
                if i % ddp_world_size != ddp_rank:
                    continue
                _, tokens, mask, label = render_example(example)
                tokens, mask = tokens.to(device), mask.to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(tokens)
                    _, pred_norm, _, _ = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            # reduce the stats across all processes
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if ddp_rank == 0:
                print(f'HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}')
                with open(log_file, 'a') as f:
                    f.write(f'{step} hella {acc_norm:.4f}\n')

        # generate some sample
        if (step % 250 == 0 or last_step) and (not use_compile):
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = enc.encode("Hello, I'm a language model,")  # (8,)
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (B,8)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < max_length:
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(xgen)
                    logits = logits[:, -1, :]  # (B, vocab_size)
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # (B,50)
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                    # pick value from topk_indices according to ix, along last dim=-1
                    xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                    # append to the sequences
                    xgen = torch.cat((xgen, xcol), dim=1)
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f'rank {ddp_rank} sample {i}: {decoded}')

        # training
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            # We have to scale the loss to account for gradient accumulation.
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        dt = t1 - t0

        tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / dt

        if ddp_rank == 0:
            print(
                f'step {step}, loss: {loss_accum.item():.6f}, lr: {lr:.4e}, norm: {norm:.4f}, time: {dt * 1000:.2f}ms, tok/s: {tokens_per_sec:.2f}')
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    if ddp:
        destroy_process_group()
