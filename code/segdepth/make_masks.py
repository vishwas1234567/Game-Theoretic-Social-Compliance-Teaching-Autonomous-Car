import sys, json, cv2, numpy as np
sys.path.insert(0, 'code/segdepth'); from annotations import A
V = '/mnt/user-data/attachments/250c89e9-e5fc-531a-bd79-74f1bc20830f/4d53d841-vidssave.com_Night_Data_Self_Driving_Car_Indian_Streets_1_360P.mp4'
CLS = ['road', 'sidewalk', 'building/veg.', 'sky', 'dynamic']; COL = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [70, 130, 180], [220, 20, 60]], np.uint8)
cap = cv2.VideoCapture(V); times = sorted(A); imgs, masks = [], []
for t in times:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * 30)); ok, fr = cap.read(); fr = cv2.resize(fr, (320, 180), interpolation=cv2.INTER_AREA)
    m = np.full((180, 320), 2, np.uint8); a = A[t]
    for p in a['sky']: cv2.fillPoly(m, [np.array(p, np.int32)], 3)
    cv2.fillPoly(m, [np.array(a['road'], np.int32)], 0)
    for p in a['side']: cv2.fillPoly(m, [np.array(p, np.int32)], 1)
    for x0, y0, x1, y1 in a['dyn']: m[y0:y1, x0:x1] = 4
    m[128:] = 255
    imgs.append(fr); masks.append(m)
np.savez_compressed('data/segdepth/seg_labels.npz', img=np.array(imgs), mask=np.array(masks), times=np.array(times))
# QA montage
tiles = []
for fr, m in list(zip(imgs, masks))[:24]:
    ov = fr.copy(); c = COL[np.minimum(m, 4)]; ov = np.where((m == 255)[..., None], fr, (0.5 * fr + 0.5 * c).astype(np.uint8)); tiles.append(cv2.resize(ov, (213, 120)))
rows = [np.hstack(tiles[i:i + 6]) for i in range(0, 24, 6)]
cv2.imwrite('data/segdepth/ann_qa.png', np.vstack(rows)); print(len(times), 'class px frac', np.bincount(np.array(masks)[np.array(masks) < 255].ravel(), minlength=5) / (np.array(masks) < 255).sum())
