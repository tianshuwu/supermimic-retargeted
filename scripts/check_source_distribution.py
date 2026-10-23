"""Validate Python source completeness using Git's index, never ignored local files.

Run with the project's environment before publishing:
  uv run python scripts/check_source_distribution.py
"""
import json,os,subprocess,sys,tarfile,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'))
python_sources={str(p.relative_to(ROOT)) for folder in ['src','scripts','tests'] for p in (ROOT/folder).rglob('*.py')}
missing=sorted(python_sources-tracked)
if missing:raise SystemExit('Python sources missing from Git index: '+', '.join(missing))
required={'src/supermimic/data/'+name+'.py' for name in ['__init__','motion_convert','replay_buffer','recorder','sampling']}
required|={'scripts/data/'+name+'.py' for name in ['build_library','csv_to_npz_batch','filter_bones_seed']}
if not required<=tracked:raise SystemExit('Required source files missing: '+', '.join(sorted(required-tracked)))
tree=subprocess.check_output(['git','write-tree'],cwd=ROOT,text=True).strip()
with tempfile.TemporaryDirectory(prefix='supermimic-source-check-') as temp:
 temp=Path(temp);archive=temp/'source.tar'
 with archive.open('wb') as f:subprocess.run(['git','archive','--format=tar',tree],cwd=ROOT,stdout=f,check=True)
 extracted=temp/'checkout';extracted.mkdir()
 with tarfile.open(archive) as package:package.extractall(extracted,filter='data')
 env=os.environ.copy();env['PYTHONPATH']=str(extracted/'src');env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1';env.pop('SM_ROBOT_SPEC',None);env['SM_ROBOT']='g1'
 code="import importlib,pathlib,supermimic; assert pathlib.Path(supermimic.__file__).resolve().is_relative_to(pathlib.Path('src').resolve()); [importlib.import_module('supermimic.data.'+name) for name in ['motion_convert','replay_buffer','recorder','sampling']]"
 subprocess.run([sys.executable,'-c',code],cwd=extracted,env=env,check=True,timeout=90)
 for script in ['scripts/data/build_library.py','scripts/data/filter_bones_seed.py']:
  subprocess.run([sys.executable,script,'--help'],cwd=extracted,env=env,check=True,stdout=subprocess.DEVNULL,timeout=90)
 result=subprocess.run([sys.executable,'-m','pytest','tests/test_g1_fk.py','tests/test_source_tracking.py','-q'],cwd=extracted,env=env,text=True,capture_output=True,timeout=180)
 print(result.stdout);print(result.stderr)
 if result.returncode:raise SystemExit(result.returncode)
print(json.dumps({'source_tree':tree,'python_files_tracked':len(python_sources),'required_data_sources':len(required),'clean_git_archive_imports_and_clis':'passed'}))
