#!/usr/bin/env bash
# usage: ./run_all.sh path/to/video.mp4
set -e
V="$1"; mkdir -p data figures report/tables
python3 code/perception.py "$V" data/p0.csv 0 3733 &
python3 code/perception.py "$V" data/p1.csv 3734 7465; wait
python3 - <<'PY'
import pandas as pd
a=pd.read_csv('data/p0.csv'); b=pd.read_csv('data/p1.csv')
pd.concat([a,b]).sort_values('frame').to_csv('data/perception.csv',index=False)
PY
python3 code/video_analysis.py
python3 code/evaluate.py 400
python3 code/viz_detections.py "$V"
python3 code/odd_features.py "$V" data/o0.csv 0 3733 &
python3 code/odd_features.py "$V" data/o1.csv 3734 7465; wait
python3 -c "import pandas as pd;pd.concat([pd.read_csv('data/o0.csv'),pd.read_csv('data/o1.csv')]).sort_values('t').to_csv('data/odd_features.csv',index=False)"
python3 code/odd_experiment.py 300
V2="$V"
python3 code/segdepth/data_prep.py "$V2" data/segdepth/depth_a.npz 1470 4480 &
python3 code/segdepth/data_prep.py "$V2" data/segdepth/depth_b.npz 4480 7460; wait
python3 code/segdepth/make_masks.py
python3 code/segdepth/unet_np.py            # gradient check + speed
python3 code/segdepth/train.py 1800         # ~30 min on CPU
python3 code/segdepth/evaluate_segdepth.py
cd report && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
