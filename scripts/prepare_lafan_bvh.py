"""Select the shared LAFAN clips and rename skeleton joints simultaneously.

Follow with align_lafan_bone_frames.py --pelvis-pivot before SOMA retargeting.
"""
import argparse,csv,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
config=json.loads((ROOT/'configs/lafan-source-adapter.json').read_text());names=config['selected_clips'];mapping=config['joint_renames']
files={p.stem:p for p in a.input.rglob('*.bvh')};missing=set(names)-files.keys()
if missing:raise SystemExit(f'Missing selected LAFAN clips: {sorted(missing)}')
a.output.mkdir(parents=True,exist_ok=True)
for name in names:
 text=files[name].read_text()
 text=re.sub(r'(?m)^(\s*(?:ROOT|JOINT)\s+)(\S+)',lambda m:m[1]+mapping.get(m[2],m[2]),text)
 (a.output/(name+'.bvh')).write_text(text)
print('Prepared',len(names),'clips. Run bone-frame and pelvis-pivot alignment next.')
