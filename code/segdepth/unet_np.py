"""Framework-free multitask U-Net (NumPy, manual backprop). PyTorch is unavailable in this workspace.
Input (B,3,72,128). Heads: 5-class segmentation logits and 1-channel log-depth.
Encoder 8-16-32, bottleneck 48, decoder with skip connections (concat), 3x3 convs + ReLU, 2x2 max-pool, nearest up-sampling."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv


class Conv:
    def __init__(s, ci, co, k, rng):
        s.k = k; s.W = (rng.standard_normal((co, ci * k * k)) * np.sqrt(2.0 / (ci * k * k))).astype(np.float32); s.b = np.zeros(co, np.float32)
        s.gW = np.zeros_like(s.W); s.gb = np.zeros_like(s.b); s.ci, s.co = ci, co

    def fwd(s, x):
        B, C, H, W = x.shape; k = s.k; p = k // 2
        if k == 1: cols = x.transpose(0, 2, 3, 1).reshape(-1, C)
        else:
            xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
            cols = swv(xp, (k, k), axis=(2, 3)).transpose(0, 2, 3, 1, 4, 5).reshape(B * H * W, C * k * k)
        s.cols, s.shape = cols, x.shape
        return (cols @ s.W.T + s.b).reshape(B, H, W, s.co).transpose(0, 3, 1, 2)

    def bwd(s, d):
        B, C, H, W = s.shape; k = s.k; p = k // 2
        dm = np.ascontiguousarray(d.transpose(0, 2, 3, 1)).reshape(-1, s.co)
        s.gW += dm.T @ s.cols; s.gb += dm.sum(0)
        dc = dm @ s.W
        if k == 1: return dc.reshape(B, H, W, C).transpose(0, 3, 1, 2)
        dc = dc.reshape(B, H, W, C, k, k); dxp = np.zeros((B, C, H + 2 * p, W + 2 * p), np.float32)
        for i in range(k):
            for j in range(k): dxp[:, :, i:i + H, j:j + W] += dc[:, :, :, :, i, j].transpose(0, 3, 1, 2)
        return dxp[:, :, p:p + H, p:p + W]


def pool_f(x):
    B, C, H, W = x.shape; r = x.reshape(B, C, H // 2, 2, W // 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(B, C, H // 2, W // 2, 4)
    idx = r.argmax(-1); return np.take_along_axis(r, idx[..., None], -1)[..., 0], idx


def pool_b(d, idx, shape):
    B, C, H, W = shape; g = np.zeros((B, C, H // 2, W // 2, 4), np.float32); np.put_along_axis(g, idx[..., None], d[..., None], -1)
    return g.reshape(B, C, H // 2, W // 2, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)


up_f = lambda x: x.repeat(2, 2).repeat(2, 3)
up_b = lambda d: d.reshape(d.shape[0], d.shape[1], d.shape[2] // 2, 2, d.shape[3] // 2, 2).sum((3, 5))


class UNet:
    def __init__(s, seed=0, c=(8, 16, 32, 48), n_cls=5):
        r = np.random.default_rng(seed); a, b, cc, d = c
        s.L = dict(e1a=Conv(3, a, 3, r), e1b=Conv(a, a, 3, r), e2a=Conv(a, b, 3, r), e2b=Conv(b, b, 3, r), e3a=Conv(b, cc, 3, r), e3b=Conv(cc, cc, 3, r),
                   ba=Conv(cc, d, 3, r), bb=Conv(d, d, 3, r), d3a=Conv(d + cc, cc, 3, r), d3b=Conv(cc, cc, 3, r), d2a=Conv(cc + b, b, 3, r), d2b=Conv(b, b, 3, r),
                   d1a=Conv(b + a, a, 3, r), d1b=Conv(a, a, 3, r), seg=Conv(a, n_cls, 1, r), dep=Conv(a, 1, 1, r))
        s.c = c

    def params(s):
        return [(l.W, l.gW) for l in s.L.values()] + [(l.b, l.gb) for l in s.L.values()]

    def zero(s):
        for l in s.L.values(): l.gW[:] = 0; l.gb[:] = 0

    def _cr(s, name, x):
        y = s.L[name].fwd(x); m = y > 0; s.cache[name] = m; return y * m

    def _crb(s, name, d): return s.L[name].bwd(d * s.cache[name])

    def forward(s, x):
        s.cache = {}; L = s._cr
        e1 = L('e1b', L('e1a', x)); p1, i1 = pool_f(e1)
        e2 = L('e2b', L('e2a', p1)); p2, i2 = pool_f(e2)
        e3 = L('e3b', L('e3a', p2)); p3, i3 = pool_f(e3)
        bt = L('bb', L('ba', p3))
        u3 = np.concatenate([up_f(bt), e3], 1); d3 = L('d3b', L('d3a', u3))
        u2 = np.concatenate([up_f(d3), e2], 1); d2 = L('d2b', L('d2a', u2))
        u1 = np.concatenate([up_f(d2), e1], 1); d1 = L('d1b', L('d1a', u1))
        s.sh = (e1.shape, e2.shape, e3.shape, bt.shape, d3.shape, d2.shape); s.idx = (i1, i2, i3)
        return s.L['seg'].fwd(d1), s.L['dep'].fwd(d1)[:, 0], d1

    def backward(s, dseg, ddep):
        L = s._crb; B = s
        d1 = np.zeros(s.sh[0][:1] + (s.c[0],) + s.sh[0][2:], np.float32)
        if dseg is not None: d1 = d1 + s.L['seg'].bwd(dseg)
        if ddep is not None: d1 = d1 + s.L['dep'].bwd(ddep[:, None])
        g = L('d1a', L('d1b', d1)); a = s.c[0]; b = s.c[1]; cc = s.c[2]; d = s.c[3]
        g_up2, g_e1 = g[:, :b], g[:, b:]
        g = L('d2a', L('d2b', up_b(g_up2))); g_up3, g_e2 = g[:, :cc], g[:, cc:]
        g = L('d3a', L('d3b', up_b(g_up3))); g_bt, g_e3 = g[:, :d], g[:, d:]
        g = up_b(g_bt); g = L('ba', L('bb', g))
        g = pool_b(g, s.idx[2], s.sh[2]) + g_e3; g = L('e3a', L('e3b', g))
        g = pool_b(g, s.idx[1], s.sh[1]) + g_e2; g = L('e2a', L('e2b', g))
        g = pool_b(g, s.idx[0], s.sh[0]) + g_e1; L('e1a', L('e1b', g))


class Adam:
    def __init__(s, params, lr=2e-3, b1=.9, b2=.999):
        s.p = params; s.m = [np.zeros_like(p) for p, _ in params]; s.v = [np.zeros_like(p) for p, _ in params]; s.t = 0; s.lr, s.b1, s.b2 = lr, b1, b2

    def step(s, clip=5.0):
        s.t += 1; tot = np.sqrt(sum(float((g ** 2).sum()) for _, g in s.p)); sc = min(1.0, clip / (tot + 1e-9))
        for i, (p, g) in enumerate(s.p):
            g = g * sc; s.m[i] = s.b1 * s.m[i] + (1 - s.b1) * g; s.v[i] = s.b2 * s.v[i] + (1 - s.b2) * g * g
            p -= s.lr * (s.m[i] / (1 - s.b1 ** s.t)) / (np.sqrt(s.v[i] / (1 - s.b2 ** s.t)) + 1e-8)
        return tot


def seg_loss(logits, y, w):
    """masked weighted CE; y (B,H,W) int with 255 ignored. returns loss, dlogits"""
    z = logits - logits.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    m = (y != 255); yy = np.where(m, y, 0); n = max(m.sum(), 1)
    pt = np.take_along_axis(p, yy[:, None], 1)[:, 0]; wt = w[yy] * m
    loss = -(wt * np.log(pt + 1e-9)).sum() / n
    g = p.copy(); np.put_along_axis(g, yy[:, None], np.take_along_axis(g, yy[:, None], 1) - 1, 1)
    return float(loss), (g * wt[:, None] / n).astype(np.float32)


def depth_loss(pred, t, valid, delta=0.3):
    """Huber on log-depth, masked."""
    n = max(valid.sum(), 1); e = (pred - t) * valid; a = np.abs(e)
    loss = np.where(a < delta, .5 * e * e, delta * (a - .5 * delta)).sum() / n
    g = np.where(a < delta, e, delta * np.sign(e)) / n
    return float(loss), g.astype(np.float32)


if __name__ == '__main__':
    import time
    rng = np.random.default_rng(1); net = UNet(0)
    # gradient check on a few random weights (small input)
    x = rng.standard_normal((2, 3, 16, 24)).astype(np.float32); y = rng.integers(0, 5, (2, 16, 24)); w = np.ones(5, np.float32)
    td = rng.standard_normal((2, 16, 24)).astype(np.float32); vd = (rng.random((2, 16, 24)) > .5).astype(np.float32)
    def total():
        s, d, _ = net.forward(x); return seg_loss(s, y, w)[0] + depth_loss(d, td, vd)[0]
    net.zero(); s, d, _ = net.forward(x); _, gs = seg_loss(s, y, w); _, gd = depth_loss(d, td, vd); net.backward(gs, gd)
    worst = 0
    for name in ['e1a', 'e2b', 'ba', 'd3a', 'd1b', 'seg', 'dep']:
        l = net.L[name]
        for _ in range(3):
            i = (rng.integers(0, l.W.shape[0]), rng.integers(0, l.W.shape[1])); old = l.W[i]; eps = 1e-2
            l.W[i] = old + eps; a = total(); l.W[i] = old - eps; b = total(); l.W[i] = old
            num = (a - b) / (2 * eps); ana = l.gW[i]; worst = max(worst, abs(num - ana) / (abs(num) + abs(ana) + 1e-6))
    print('gradient check worst relative error: %.4f' % worst)
    xb = rng.standard_normal((8, 3, 72, 128)).astype(np.float32); t0 = time.time()
    for _ in range(2):
        net.zero(); s, d, _ = net.forward(xb); net.backward(np.ones_like(s) * 1e-3, np.ones_like(d) * 1e-3)
    print('train step (batch 8) %.2fs' % ((time.time() - t0) / 2))
