import json
import torch
import torch.distributed as dist


def gather_floats(local: list[float], device) -> list[float]:
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return local
    t = torch.tensor(local, device=device)
    size = torch.tensor(len(local), device=device)
    sizes = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, size)
    max_size = max(s.item() for s in sizes)
    buf = torch.zeros(max_size, device=device)
    buf[:len(local)] = t
    bufs = [torch.zeros(max_size, device=device) for _ in range(dist.get_world_size())]
    dist.all_gather(bufs, buf)
    merged = []
    for s, b in zip(sizes, bufs):
        merged.extend(b[:s.item()].cpu().tolist())
    return merged


def gather_string_lists(local: list[list[dict]], device) -> list[list[dict]]:
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return local
    payload = json.dumps(local).encode()
    size = torch.tensor(len(payload), device=device)
    sizes = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, size)
    max_size = max(s.item() for s in sizes)
    buf = torch.zeros(max_size, dtype=torch.uint8, device=device)
    buf[:len(payload)] = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    bufs = [torch.zeros(max_size, dtype=torch.uint8, device=device) for _ in range(dist.get_world_size())]
    dist.all_gather(bufs, buf)
    merged = []
    for s, b in zip(sizes, bufs):
        merged.extend(json.loads(bytes(b[:s.item()].cpu().tolist()).decode()))
    return merged
