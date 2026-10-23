"""Retain validated LAFAN curriculum after BONES QC and emit accepted clip names."""
import argparse,csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--robot',choices=['g1','a3','x2_ultra'],required=True);p.add_argument('--count',type=int,required=True);a=p.parse_args()
dest=ROOT/f'data/npz/{a.robot}/bones_{a.count}';manifest=dest/'manifest_filtered.csv'
rows={r['name']:r for r in csv.DictReader(manifest.open())}
lafan=ROOT/f'data/npz/{a.robot}/lafan1';files=list(lafan.glob('*.npz'))
if len(files)!=40:raise SystemExit('Expected the 40 validated LAFAN clips before expanding.')
for src in files:
 with np.load(src) as data:frames=len(data['joint_pos'])
 target=dest/src.name
 if not target.exists():target.symlink_to(src.resolve())
 rows[src.stem]={'name':src.stem,'frames':frames,'keep':'1','reason':''}
accepted=[name for name,row in rows.items() if row['keep']=='1']
for name in accepted:
 if not (dest/(name+'.npz')).exists():raise SystemExit('Accepted NPZ missing: '+name)
with manifest.open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=['name','frames','keep','reason']);writer.writeheader();writer.writerows(rows.values())
(dest/'clips.txt').write_text('\n'.join(sorted(accepted))+'\n')
print(json.dumps({'robot':a.robot,'accepted_bones_clips':len(accepted)-40,'retained_lafan_clips':40}))
