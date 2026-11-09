import sys, json, time, numpy as np, cv2, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter, uniform_filter
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
sys.path.insert(0, 'code/segdepth')
from common import *; import unet_np as U
plt.rcParams.update({'font.size': 8, 'figure.dpi': 150})
CLS = ['road', 'sidewalk', 'building/veg.', 'sky', 'dynamic']
PAL = np.array([[128, 64, 128], [244, 35, 232], [90, 90, 90], [70, 130, 180], [220, 20, 60]], np.uint8)
VID = '/mnt/user-data/attachments/250c89e9-e5fc-531a-bd79-74f1bc20830f/4d53d841-vidssave.com_Night_Data_Self_Driving_Car_Indian_Streets_1_360P.mp4'
DOFF = 2.5; HB = 47                                        # rows >= HB are bonnet (ignored)
D = load_data(); R = {}
net = U.UNet(0); w = np.load('data/segdepth/unet_weights.npz')
for n, l in net.L.items(): l.W[:] = w[f'{n}_W']; l.b[:] = w[f'{n}_b']


def predict(imgs, bs=16):
    S, Z = [], []
    for i in range(0, len(imgs), bs):
        s, d, _ = net.forward(prep(imgs[i:i + bs])); S.append(s); Z.append(np.exp(d + DOFF))
    return np.concatenate(S), np.concatenate(Z)


# ======================= SEGMENTATION =======================
tr, te = np.where(D['s_train'])[0], np.where(D['s_test'])[0]
Yte = D['smask'][te]; Xte = D['simg'][te]
seg_u, _ = predict(Xte); P_u = seg_u.argmax(1)
# baseline 1: position prior
cnt = np.zeros((5, 72, 128))
for k in tr:
    for c in range(5): cnt[c] += (D['smask'][k] == c)
P_prior = np.repeat(cnt.argmax(0)[None], len(te), 0)


def feats(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32); hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    yy, xx = np.mgrid[0:72, 0:128].astype(np.float32); gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    m5 = uniform_filter(g, 5); sd5 = np.sqrt(np.maximum(uniform_filter(g * g, 5) - m5 ** 2, 0))
    f = [xx / 128, yy / 72, *[img[..., c] / 255. for c in range(3)], hsv[..., 1] / 255, gaussian_filter(g, 2) / 255, gaussian_filter(g, 6) / 255, sd5 / 64, np.hypot(gx, gy) / 255,
         g.mean(1, keepdims=True).repeat(128, 1) / 255]
    return np.stack(f, -1).reshape(-1, len(f))


Ftr = np.concatenate([feats(D['simg'][k]) for k in tr]); Ytr = np.concatenate([D['smask'][k].ravel() for k in tr]); keep = Ytr != 255
t0 = time.time(); gb = HistGradientBoostingClassifier(max_iter=120, max_depth=6, learning_rate=0.1, random_state=0).fit(Ftr[keep], Ytr[keep])
P_gb = np.array([gb.predict(feats(im)).reshape(72, 128) for im in Xte]); print('gb seg fit', time.time() - t0)


def seg_metrics(P, Y):
    out = {}; v = Y != 255
    out['pixacc'] = float((P[v] == Y[v]).mean()); ious = []
    for c in range(5):
        inter = ((P == c) & (Y == c) & v).sum(); uni = (((P == c) | (Y == c)) & v).sum()
        out[f'iou_{c}'] = float(inter / uni) if (Y[v] == c).any() else None
        if out[f'iou_{c}'] is not None: ious.append(out[f'iou_{c}'])
    out['miou'] = float(np.mean(ious)); return out


rng = np.random.default_rng(0)
def boot_miou(P, Y, B=1000):
    vals = []
    for _ in range(B):
        ii = rng.integers(0, len(Y), len(Y)); vals.append(seg_metrics(P[ii], Y[ii])['miou'])
    return np.percentile(vals, [2.5, 97.5]).tolist()


segres = {}
for name, P in [('Position prior', P_prior), ('Gradient boosting (per pixel)', P_gb), ('U-Net (NumPy, ours)', P_u)]:
    m = seg_metrics(P, Yte); m['miou_ci'] = boot_miou(P, Yte); segres[name] = m
R['seg'] = segres
conf = np.zeros((5, 5)); v = Yte != 255
for a in range(5):
    for b in range(5): conf[a, b] = ((Yte == a) & (P_u == b) & v).sum()
conf_n = conf / np.maximum(conf.sum(1, keepdims=True), 1)
rows = []
for name, m in segres.items():
    rows.append([name, f"{100 * m['pixacc']:.1f}"] + [('--' if m[f'iou_{c}'] is None else f"{100 * m[f'iou_{c}']:.1f}") for c in range(5)] + [f"{100 * m['miou']:.1f} [{100 * m['miou_ci'][0]:.1f},{100 * m['miou_ci'][1]:.1f}]"])
pd.DataFrame(rows, columns=['Method', 'Pixel acc.'] + [c.replace('/', '/') for c in CLS] + ['mIoU [95\\% CI]']).to_latex('report/tables/seg_results.tex', index=False, escape=False, column_format='lrrrrrrr')

# qualitative
fig, ax = plt.subplots(4, 5, figsize=(7.4, 5.0))
sel = [0, 2, 4, 6]
for r_, k in enumerate(sel):
    col = lambda P: np.where((Yte[k] == 255)[..., None], 0, PAL[np.minimum(P, 4)])
    for c_, (im, tt) in enumerate([(Xte[k], 'frame'), (col(Yte[k]), 'hand label'), (col(P_prior[k]), 'position prior'), (col(P_gb[k]), 'grad. boosting'), (col(P_u[k]), 'U-Net')]):
        ax[r_, c_].imshow(im); ax[r_, c_].axis('off')
        if r_ == 0: ax[r_, c_].set_title(tt, fontsize=7)
    ax[r_, 0].text(2, 8, f"t={int(D['times'][te][k])}s", color='w', fontsize=6)
handles = [plt.Rectangle((0, 0), 1, 1, color=PAL[i] / 255) for i in range(5)]; fig.legend(handles, CLS, ncol=5, loc='lower center', fontsize=7, frameon=False)
plt.tight_layout(rect=(0, .04, 1, 1)); plt.savefig('figures/seg_qualitative.png'); plt.close()
fig, ax = plt.subplots(figsize=(3.4, 3.0)); ax.imshow(conf_n, cmap='Blues', vmin=0, vmax=1); ax.set_xticks(range(5)); ax.set_xticklabels(CLS, rotation=35, ha='right'); ax.set_yticks(range(5)); ax.set_yticklabels(CLS)
for a in range(5):
    for b in range(5): ax.text(b, a, f"{conf_n[a, b]:.2f}", ha='center', va='center', fontsize=6, color='w' if conf_n[a, b] > .5 else 'k')
ax.set_xlabel('predicted'); ax.set_ylabel('hand label'); ax.set_title('U-Net confusion (row-normalised)', fontsize=8); plt.tight_layout(); plt.savefig('figures/seg_confusion.png'); plt.close()

# ======================= DEPTH =======================
dte = np.where(D['d_test'])[0]; dtr = np.where(D['d_train'])[0]
Xd = D['dimg'][dte]; Zt = D['dZ'][dte]; vm = np.isfinite(Zt); vm[:, HB:] = False
_, Z_u = predict(Xd)
# baselines
logtr = np.log(D['dZ'][dtr]); const = float(np.exp(np.nanmedian(logtr)))
rowprior = np.exp(np.array([np.nanmedian(logtr[:, r_, :]) if np.isfinite(logtr[:, r_, :]).any() else np.nan for r_ in range(72)])); rowprior = pd.Series(rowprior).interpolate(limit_direction='both').values
yy = np.arange(72, dtype=np.float32)[:, None]; fpx = 64 / np.tan(np.deg2rad(35)); yh = 0.40 * 72
Zg = np.where(yy > yh + 1.5, fpx * 1.2 / np.maximum(yy - yh, 1.5), 60.0).clip(3, 60) * np.ones((1, 128))
Fd = np.concatenate([feats(D['dimg'][k]) for k in dtr[::2]]); Td = np.concatenate([np.log(D['dZ'][k]).ravel() for k in dtr[::2]]); kd = np.isfinite(Td) & (np.tile(np.repeat(np.arange(72) < HB, 128), len(dtr[::2])))
t0 = time.time(); gbr = HistGradientBoostingRegressor(max_iter=120, max_depth=6, random_state=0).fit(Fd[kd], Td[kd]); print('gb depth fit', time.time() - t0)
Z_gb = np.array([np.exp(gbr.predict(feats(im)).reshape(72, 128)) for im in Xd])
preds = {'Constant (train median)': np.full_like(Z_u, const), 'Row prior': np.repeat(rowprior[None, :, None], len(dte), 0) * np.ones((1, 1, 128)), 'Ground-plane geometry': np.repeat(Zg[None], len(dte), 0),
         'Gradient boosting (per pixel)': Z_gb, 'U-Net (NumPy, ours)': Z_u}


def dmetrics(Zp, Zt, vm, align=False):
    out = {}; a, b = Zp[vm], Zt[vm]
    if align:                                         # per-image median scale alignment
        a = np.concatenate([Zp[i][vm[i]] * (np.median(Zt[i][vm[i]]) / np.median(Zp[i][vm[i]])) for i in range(len(Zp)) if vm[i].any()]); b = np.concatenate([Zt[i][vm[i]] for i in range(len(Zp)) if vm[i].any()])
    e = np.log(a) - np.log(b)
    out['absrel'] = float(np.mean(np.abs(a - b) / b)); out['rmse_log'] = float(np.sqrt(np.mean(e ** 2))); out['d1'] = float(np.mean(np.maximum(a / b, b / a) < 1.25))
    out['silog'] = float(np.sqrt(np.mean(e ** 2) - np.mean(e) ** 2)); out['rho'] = float(spearmanr(a, b).statistic); return out


dres = {}; blocks = (D['meta'][dte][:, 1] // 20).astype(int)
for name, Zp in preds.items():
    m = dmetrics(Zp, Zt, vm); m['aligned'] = dmetrics(Zp, Zt, vm, True)
    m['per_block_absrel'] = [dmetrics(Zp[blocks == b], Zt[blocks == b], vm[blocks == b])['absrel'] for b in np.unique(blocks)]
    dres[name] = m
R['depth'] = dres; R['depth_n'] = dict(test_frames=int(len(dte)), valid_px=int(vm.sum()), train_frames=int(len(dtr)), blocks=int(len(np.unique(blocks))))
rows = [[n, f"{m['absrel']:.3f}", f"{m['rmse_log']:.3f}", f"{100 * m['d1']:.1f}", f"{m['aligned']['absrel']:.3f}", f"{m['silog']:.3f}", f"{m['rho']:.2f}"] for n, m in dres.items()]
pd.DataFrame(rows, columns=['Method', 'AbsRel', 'RMSE-log', '$\\delta<1.25$ (\\%)', 'AbsRel (scale-aligned)', 'SILog', 'Spearman $\\rho$']).replace('nan', '--').to_latex('report/tables/depth_results.tex', index=False, escape=False, column_format='lrrrrrr')
# error by distance bin and by predicted class
seg_d, _ = predict(Xd); Pd = seg_d.argmax(1)
bins = [(3, 10), (10, 20), (20, 35), (35, 70)]; dist_rows = []
for lo, hi in bins:
    mm = vm & (Zt >= lo) & (Zt < hi); dist_rows.append([f'{lo}--{hi} m', int(mm.sum())] + [f"{np.mean(np.abs(preds[n][mm] - Zt[mm]) / Zt[mm]):.3f}" for n in ['Row prior', 'Gradient boosting (per pixel)', 'U-Net (NumPy, ours)']])
pd.DataFrame(dist_rows, columns=['Label depth', 'Valid px', 'Row prior', 'Grad. boosting', 'U-Net']).to_latex('report/tables/depth_bins.tex', index=False, escape=False, column_format='lrrrr')
cls_rows = []
for c in [0, 1, 2, 4]:
    mm = vm & (Pd == c)
    if mm.sum() > 200: cls_rows.append([CLS[c], int(mm.sum()), f"{np.mean(np.abs(Z_u[mm] - Zt[mm]) / Zt[mm]):.3f}", f"{np.mean(np.abs(preds['Ground-plane geometry'][mm] - Zt[mm]) / Zt[mm]):.3f}"])
pd.DataFrame(cls_rows, columns=['Predicted class', 'Valid px', 'U-Net AbsRel', 'Ground-plane AbsRel']).to_latex('report/tables/depth_by_class.tex', index=False, escape=False, column_format='lrrr')

# ---- lamp-range cross-check
cap = cv2.VideoCapture(VID); f320 = 160 / np.tan(np.deg2rad(35)); lamp = []
for j, k in enumerate(dte):
    f = int(D['meta'][k][0]); cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, fr = cap.read()
    if not ok: continue
    s = cv2.resize(fr, (320, 180), interpolation=cv2.INTER_AREA); hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
    red = (((hsv[..., 0] < 10) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 90) & (hsv[..., 2] > 150)).astype(np.uint8); red[:90 - 30] = 0; red[119:] = 0
    n, _, st, cen = cv2.connectedComponentsWithStats(cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))); rr = [(cen[i], st[i]) for i in range(1, n) if 3 <= st[i, 4] <= 400]
    best = None
    for a in range(len(rr)):
        for b in range(a + 1, len(rr)):
            (c1, s1), (c2, s2) = rr[a], rr[b]
            if abs(c1[1] - c2[1]) < 4 and 7 <= abs(c1[0] - c2[0]) <= 70 and 0.5 < s1[4] / max(1, s2[4]) < 2.0:
                sep = abs(c1[0] - c2[0])
                if best is None or sep > best[0]: best = (sep, (c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2)
    if best is None: continue
    sep, cx, cy = best; Zl = f320 * 1.5 / sep
    x0, x1 = int((cx - 0.9 * sep) * .4), int((cx + 0.9 * sep) * .4) + 1; y0, y1 = int(cy * .4), int((cy + 0.6 * sep) * .4) + 1
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, 128), min(y1, 72)
    if x1 - x0 < 2 or y1 - y0 < 1 or not (3 < Zl < 60): continue
    box = (slice(y0, y1), slice(x0, x1)); mv = vm[j][box]
    lamp.append(dict(frame=f, Zl=Zl, unet=float(np.median(Z_u[j][box])), row=float(np.median(preds['Row prior'][j][box])), gb=float(np.median(Z_gb[j][box])), const=const,
                     motion=float(np.median(Zt[j][box][mv])) if mv.sum() > 3 else np.nan, mfrac=float(mv.mean())))
L = pd.DataFrame(lamp); L.to_csv('data/segdepth/lamp_check.csv', index=False)
lr = {}
if len(L) >= 5:
    for k in ['unet', 'row', 'gb', 'const', 'motion']:
        d = L.dropna(subset=[k]) if k == 'motion' else L
        lr[k] = dict(n=int(len(d)), medAbsRel=float(np.median(np.abs(d[k] - d.Zl) / d.Zl)), rho=float(spearmanr(d[k], d.Zl).statistic) if d[k].nunique() > 2 else None)
R['lamp'] = lr; R['lamp_n'] = int(len(L))

# ---- figures: depth qualitative
cm = plt.get_cmap('turbo'); vmin, vmax = 3, 60
def dcol(Z, mask=None):
    z = (np.log(np.clip(Z, vmin, vmax)) - np.log(vmin)) / (np.log(vmax) - np.log(vmin)); im = cm(1 - z)[..., :3]
    if mask is not None: im = np.where(mask[..., None], im, 0.1)
    im[HB:] = 0.15; return im
fig, ax = plt.subplots(4, 4, figsize=(7.4, 4.2)); sel = np.linspace(5, len(dte) - 5, 4).astype(int)
for r_, j in enumerate(sel):
    for c_, (im, tt) in enumerate([(Xd[j], 'frame'), (dcol(Zt[j], vm[j]), 'motion-parallax label'), (dcol(Zg), 'ground-plane geometry'), (dcol(Z_u[j]), 'U-Net (single frame)')]):
        ax[r_, c_].imshow(im); ax[r_, c_].axis('off')
        if r_ == 0: ax[r_, c_].set_title(tt, fontsize=7)
    ax[r_, 0].text(2, 8, f"t={D['meta'][dte][j][1]:.0f}s", color='w', fontsize=6)
sm = plt.cm.ScalarMappable(cmap=cm.reversed(), norm=matplotlib.colors.LogNorm(vmin, vmax)); cb = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=.03, pad=.02); cb.set_ticks([3, 5, 10, 20, 40]); cb.set_ticklabels(['3', '5', '10', '20', '40']); cb.ax.minorticks_off(); cb.set_label('depth (m, assumed scale)', fontsize=7)
plt.savefig('figures/depth_qualitative.png', bbox_inches='tight'); plt.close()
# scatter + error by distance
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
a, b = Z_u[vm], Zt[vm]; sub = rng.choice(len(a), min(30000, len(a)), replace=False)
ax[0].hexbin(np.log10(b[sub]), np.log10(a[sub]), gridsize=40, cmap='viridis', mincnt=1, bins='log'); ax[0].plot([.5, 1.9], [.5, 1.9], 'r--', lw=.8); ax[0].set_xlabel('log10 label depth'); ax[0].set_ylabel('log10 U-Net depth'); ax[0].set_title('Held-out blocks: prediction vs label', fontsize=8)
for n, mk in [('Row prior', 'C1'), ('Gradient boosting (per pixel)', 'C2'), ('U-Net (NumPy, ours)', 'C0')]:
    ax[1].plot(range(len(bins)), [np.mean(np.abs(preds[n][(vm & (Zt >= lo) & (Zt < hi))] - Zt[(vm & (Zt >= lo) & (Zt < hi))]) / Zt[(vm & (Zt >= lo) & (Zt < hi))]) for lo, hi in bins], 'o-', color=mk, label=n.split(' (')[0])
ax[1].set_xticks(range(len(bins))); ax[1].set_xticklabels([f'{lo}-{hi} m' for lo, hi in bins]); ax[1].set_ylabel('AbsRel'); ax[1].legend(fontsize=6); ax[1].set_title('Error by label depth', fontsize=8); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig('figures/depth_scatter.png'); plt.close()
# lamp scatter
if len(L) >= 5:
    fig, ax = plt.subplots(figsize=(3.6, 3.0)); ax.plot(L.Zl, L.unet, 'o', ms=3, label='U-Net'); ax.plot(L.Zl, L.row, 's', ms=3, label='row prior'); mo = L.dropna(subset=['motion']); ax.plot(mo.Zl, mo.motion, '^', ms=3, label='motion parallax')
    ax.plot([3, 40], [3, 40], 'k--', lw=.7); ax.set_xlabel('tail-light range (m)'); ax.set_ylabel('depth at vehicle (m)'); ax.legend(fontsize=6); ax.set_title(f'Lead vehicles (n={len(L)})', fontsize=8); ax.grid(alpha=.3); plt.tight_layout(); plt.savefig('figures/depth_lamp.png'); plt.close()

# ---- bird's-eye view from predicted depth + segmentation
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.8)); f72 = 64 / np.tan(np.deg2rad(35))
for a_, j in zip(ax, [sel[0], sel[1], sel[3]]):
    Zp = Z_u[j]; S = Pd[j]; uu = np.arange(128)[None, :] * np.ones((72, 1)); X = (uu - 64) * Zp / f72
    for c, col in [(2, 'gray'), (1, 'm'), (0, 'tab:blue'), (4, 'r')]:
        mm = (S == c) & (np.arange(72)[:, None] < HB)
        a_.scatter(X[mm][::3], Zp[mm][::3], s=1, c=col, label=CLS[c], alpha=.5)
    a_.set_xlim(-25, 25); a_.set_ylim(0, 50); a_.set_xlabel('lateral (m)'); a_.set_title(f"t={D['meta'][dte][j][1]:.0f}s", fontsize=8); a_.grid(alpha=.3)
ax[0].set_ylabel('forward (m)'); ax[2].legend(fontsize=6, markerscale=5, loc='upper right'); plt.tight_layout(); plt.savefig('figures/bev.png'); plt.close()

# ======================= FREE SPACE vs HUMAN SPEED (all frames) =======================
Xall = D['dimg']; Sall, Zall = predict(Xall); Pall = Sall.argmax(1)
road_frac = (Pall[:, 28:HB] == 0).mean((1, 2))
corr = slice(52, 76); fs = []
for i in range(len(Xall)):
    col = (Pall[i][:HB, corr] == 0).mean(1) > 0.5                  # corridor rows that are mostly road
    r0 = np.where(col)[0]
    fs.append(float(np.median(Zall[i][r0.min():r0.min() + 3, corr])) if len(r0) else np.nan)
fs = np.array(fs); tm = D['meta'][:, 1]; vspd = D['meta'][:, 2]
ok = np.isfinite(fs); rho_fs = spearmanr(fs[ok], vspd[ok]); rho_rf = spearmanr(road_frac, vspd)
R['freespace'] = dict(n=int(ok.sum()), rho_range_speed=float(rho_fs.statistic), p_range_speed=float(rho_fs.pvalue), rho_roadfrac_speed=float(rho_rf.statistic), p_roadfrac=float(rho_rf.pvalue),
                      median_range=float(np.nanmedian(fs)), p10_range=float(np.nanpercentile(fs, 10)), mean_road_frac=float(road_frac.mean()))
blk = (tm // 10).astype(int); ub = np.unique(blk); bs = []
for _ in range(1000):
    pick = rng.choice(ub, len(ub)); ii = np.concatenate([np.where((blk == b) & ok)[0] for b in pick]); bs.append(spearmanr(fs[ii], vspd[ii]).statistic)
R['freespace']['rho_range_speed_block_ci'] = np.percentile(bs, [2.5, 97.5]).tolist()
pd.DataFrame(dict(t=tm, v=vspd, road_frac=road_frac, freespace=fs)).to_csv('data/segdepth/freespace.csv', index=False)
fig, ax = plt.subplots(3, 1, figsize=(7.2, 4.8), sharex=True)
ax[0].plot(tm, vspd * 3.6, lw=.8); ax[0].set_ylabel('ego speed (km/h)')
ax[1].plot(tm, road_frac, color='C2', lw=.8); ax[1].set_ylabel('visible road fraction')
ax[2].plot(tm, fs, '.', ms=2, color='C3'); ax[2].set_ylabel('free-space range (m)'); ax[2].set_xlabel('time (s)')
for a_ in ax: a_.grid(alpha=.3)
ax[0].set_title(f"Segmentation + depth from single frames: road visibility and free-space range (rho(range,speed)={rho_fs.statistic:.2f})", fontsize=8)
plt.tight_layout(); plt.savefig('figures/freespace_timeline.png'); plt.close()
# training curves
lg = json.load(open('data/segdepth/train_log.json')); lg = np.array(lg)
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.3)); ax[0].plot(lg[:, 0], lg[:, 1]); ax[0].set_title('segmentation loss (train batches)', fontsize=8); ax[1].plot(lg[:, 0], lg[:, 2], color='C1'); ax[1].set_title('depth Huber loss (train batches)', fontsize=8)
for a_ in ax: a_.grid(alpha=.3); a_.set_xlabel('iteration')
plt.tight_layout(); plt.savefig('figures/train_curves.png'); plt.close()
R['n_params'] = int(sum(l.W.size + l.b.size for l in net.L.values()))
json.dump(R, open('data/segdepth/results.json', 'w'), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
