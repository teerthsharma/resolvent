"""Quaternion (SU(2)) ops for Phase J Addendum L, row Q1. CPU only."""
import torch


def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product, w-first quaternions, batched over leading dims. a*b."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def qconj(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def qnormalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True)


def sequential_prefix(q: torch.Tensor) -> torch.Tensor:
    n = q.shape[-2]
    out = [q[..., 0, :]]
    for i in range(1, n):
        out.append(qmul(q[..., i, :], out[-1]))
    return torch.stack(out, dim=-2)


def prefix_scan(q: torch.Tensor) -> torch.Tensor:
    x = q.clone()
    n = x.shape[-2]
    d = 1
    while d < n:
        shifted = x[..., :-d, :]
        head = x[..., :d, :]
        combined = qmul(x[..., d:, :], shifted)
        x = torch.cat([head, combined], dim=-2)
        d *= 2
    return x
