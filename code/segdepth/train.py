import sys, time, json, numpy as np
sys.path.insert(0, 'code/segdepth')
from common import *; import unet_np as U
ITERS = int(sys.argv[1]); BS = 8; rng = np.random.default_rng(0)
D = load_data(); net = U.UNet(0); opt = U.Adam(net.params(), lr=2e-3)
si = np.where(D['s_train'])[0]; di = np.where(D['d_train'])[0]
print('seg train frames', len(si), 'test', D['s_test'].sum(), '| depth train frames', len(di), 'test', D['d_test'].sum(), flush=True)
cnt = np.bincount(D['smask'][D['s_train']][D['smask'][D['s_train']] < 255].ravel(), minlength=5).astype(np.float64); wcls = (1 / np.sqrt(cnt / cnt.sum() + 1e-3)); wcls = (wcls / wcls.mean()).astype(np.float32)
print('class weights', wcls.round(2), flush=True)
DOFF = 2.5; log = []; t0 = time.time()
for it in range(ITERS):
    lr = 2e-3 * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * it / ITERS))); opt.lr = lr
    # ---- segmentation step
    xs, ys = [], []
    for k in rng.choice(si, BS):
        a, m, _ = augment(D['simg'][k], D['smask'][k], None, rng); xs.append(a); ys.append(m)
    net.zero(); s, d, _ = net.forward(prep(np.array(xs))); ls, gs = U.seg_loss(s, np.array(ys).astype(np.int64), wcls); net.backward(gs, None); opt.step()
    # ---- depth step
    xs, zs = [], []
    for k in rng.choice(di, BS):
        a, _, z = augment(D['dimg'][k], None, D['dZ'][k], rng, geom=False); xs.append(a); zs.append(z)
    zs = np.array(zs); valid = np.isfinite(zs).astype(np.float32); tgt = np.where(valid > 0, np.log(np.where(valid > 0, zs, 1.0)) - DOFF, 0).astype(np.float32)
    net.zero(); s, d, _ = net.forward(prep(np.array(xs))); ld, gd = U.depth_loss(d, tgt, valid); net.backward(None, gd); opt.step()
    if it % 50 == 0:
        log.append((it, ls, ld)); print(f'it {it} seg {ls:.3f} depth {ld:.4f} lr {lr:.5f} {time.time()-t0:.0f}s', flush=True)
    if it % 300 == 299 or it == ITERS - 1:
        np.savez('data/segdepth/unet_weights.npz', **{f'{n}_W': l.W for n, l in net.L.items()}, **{f'{n}_b': l.b for n, l in net.L.items()})
json.dump(log, open('data/segdepth/train_log.json', 'w')); print('done', time.time() - t0, flush=True)
