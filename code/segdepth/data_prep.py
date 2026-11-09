"""Motion-parallax depth labels from the clip (monocular video + ego speed), at 128x72.
Z = dz * (1 + r/u): r = distance of a pixel from the focus of expansion (FOE), u = radial flow after derotation, dz = ego travel."""
import sys, cv2, numpy as np, pandas as pd
VID = sys.argv[1]; OUT = sys.argv[2]; F0, F1 = int(sys.argv[3]), int(sys.argv[4])
W, H = 256, 144; HOR = 0.40 * H; FOE = np.array([W / 2, HOR], np.float32)
GAP = 4                                                    # source frames between the pair (0.133 s)
P = pd.read_csv('data/perception_derived.csv')
tt, vv = P.t.values, P.v.values                            # smoothed ego speed (m/s), assumed-scale
cap = cv2.VideoCapture(VID); clahe = cv2.createCLAHE(2.0, (4, 4))
ys, xs = np.mgrid[0:H, 0:W].astype(np.float32); rx, ry = xs - FOE[0], ys - FOE[1]; r = np.hypot(rx, ry) + 1e-3
imgs, deps, meta = [], [], []
for f in range(F0, F1, 6):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, a = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, f + GAP); ok2, b = cap.read()
    if not (ok and ok2): continue
    t = f / 30.0; v = float(np.interp(t, tt, vv))
    if v < 2.0: continue
    a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA); b = cv2.resize(b, (W, H), interpolation=cv2.INTER_AREA)
    ga = clahe.apply(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)); gb = clahe.apply(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    fl = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 4, 15, 3, 5, 1.2, 0)
    band = (ys > HOR - 14) & (ys < HOR + 6)                # far field: flow ~ rotation only -> derotate
    fl[..., 0] -= np.median(fl[..., 0][band]); fl[..., 1] -= np.median(fl[..., 1][band])
    u = (fl[..., 0] * rx + fl[..., 1] * ry) / r             # radial flow
    cosang = u / (np.hypot(fl[..., 0], fl[..., 1]) + 1e-3)
    tex = np.hypot(cv2.Sobel(gb, cv2.CV_32F, 1, 0), cv2.Sobel(gb, cv2.CV_32F, 0, 1))
    dz = v * GAP / 30.0
    Z = dz * (1 + r / np.maximum(u, 1e-3))
    ok_m = (u > 0.7) & (r > 28) & (cosang > 0.8) & (tex > 30) & (ys < 0.66 * H) & (Z > 3) & (Z < 70)
    Z = np.where(ok_m, Z, np.nan).astype(np.float32)
    # block-median to 128x72
    Zb = Z.reshape(72, 2, 128, 2).transpose(0, 2, 1, 3).reshape(72, 128, 4)
    with np.errstate(all='ignore'): Zd = np.nanmedian(Zb, axis=2)
    nvalid = np.isfinite(Zd).sum()
    if nvalid < 150: continue
    imgs.append(cv2.resize(b if False else a, (128, 72), interpolation=cv2.INTER_AREA)); deps.append(Zd.astype(np.float32)); meta.append((f, t, v, nvalid))
np.savez_compressed(OUT, img=np.array(imgs), depth=np.array(deps), meta=np.array(meta))
print(OUT, len(imgs))
