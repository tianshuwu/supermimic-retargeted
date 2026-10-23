"""Resolve retargeting paths and invoke the isolated SOMA environment."""
import argparse,json,os,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--robot',choices=['a3','x2_ultra'],required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=32);a=p.parse_args()
config=json.loads((ROOT/f'configs/retarget/{a.robot}.json').read_text());config.update(import_folder=str(a.input.resolve()),export_folder=str(a.output.resolve()),batch_size=a.batch_size)
python=ROOT/'vendor/soma-retargeter/.venv/bin/python'
if not python.exists():raise SystemExit('Install the retargeting environment first; see README.')
env=os.environ.copy();env.pop('PYTHONPATH',None)
with tempfile.TemporaryDirectory(prefix='supermimic-retarget-') as temp:
 path=Path(temp)/'config.json';path.write_text(json.dumps(config))
 subprocess.run([str(python),str(ROOT/'vendor/soma-retargeter/app/bvh_to_csv_converter.py'),'--config',str(path)],cwd=ROOT,env=env,check=True)
