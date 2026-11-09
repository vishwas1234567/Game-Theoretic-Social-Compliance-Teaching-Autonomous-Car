import sys, numpy as np, cv2
sys.path.insert(0, 'code/segdepth')
MU, SD = 0.30, 0.25
def load_data():
    L = np.load('data/segdepth/seg_labels.npz'); times = L['times']
    simg = np.array([cv2.resize(i, (128, 72), interpolation=cv2.INTER_AREA) for i in L['img']])
    smask = np.array([cv2.resize(m, (128, 72), interpolation=cv2.INTER_NEAREST) for m in L['mask']])
    idx = np.arange(len(times)); s_test = idx % 3 == 2; s_train = ~s_test
    A = np.load('data/segdepth/depth_a.npz'); B = np.load('data/segdepth/depth_b.npz')
    dimg = np.concatenate([A['img'], B['img']]); dZ = np.concatenate([A['depth'], B['depth']]); meta = np.concatenate([A['meta'], B['meta']])
    t = meta[:, 1]; o = np.argsort(t); dimg, dZ, meta, t = dimg[o], dZ[o], meta[o], t[o]
    block = (t // 20).astype(int); d_test = (block % 3 == 1)
    tt = t[d_test]; dist_test = np.array([np.abs(tt - x).min() for x in t])
    seg_test_t = times[s_test]; dist_seg = np.array([np.abs(seg_test_t - x).min() for x in t])
    d_train = (~d_test) & (dist_test > 3.0) & (dist_seg > 3.0)
    return dict(simg=simg, smask=smask, times=times, s_train=s_train, s_test=s_test, dimg=dimg, dZ=dZ, meta=meta, d_train=d_train, d_test=d_test)
def prep(img): return ((img.astype(np.float32) / 255.0 - MU) / SD).transpose(0, 3, 1, 2)
def augment(img, mask=None, dep=None, rng=None, geom=True):
    """img HxWx3 uint8; random flip, scale-crop, photometric jitter."""
    H, W = img.shape[:2]
    if rng.random() < .5:
        img = img[:, ::-1]; mask = None if mask is None else mask[:, ::-1]; dep = None if dep is None else dep[:, ::-1]
    s = rng.uniform(0.75, 1.0) if geom else 1.0; h, w = int(H * s), int(W * s); y0 = rng.integers(0, H - h + 1); x0 = rng.integers(0, W - w + 1)
    img = cv2.resize(np.ascontiguousarray(img[y0:y0 + h, x0:x0 + w]), (W, H), interpolation=cv2.INTER_LINEAR)
    if mask is not None: mask = cv2.resize(np.ascontiguousarray(mask[y0:y0 + h, x0:x0 + w]), (W, H), interpolation=cv2.INTER_NEAREST)
    if dep is not None: dep = cv2.resize(np.ascontiguousarray(dep[y0:y0 + h, x0:x0 + w]), (W, H), interpolation=cv2.INTER_NEAREST)
    f = img.astype(np.float32) * rng.uniform(0.6, 1.5) ; f = 255 * (np.clip(f / 255, 0, 1) ** rng.uniform(0.8, 1.25)); f += rng.normal(0, 4, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8), mask, dep
