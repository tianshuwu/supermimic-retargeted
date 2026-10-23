"""Root data exclusions must not hide Python package or preprocessing sources."""
import subprocess
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('name,ignored',[
 ('data/motion.csv',True),
 ('src/supermimic/data/motion_convert.py',False),
 ('src/supermimic/data/replay_buffer.py',False),
 ('src/supermimic/data/recorder.py',False),
 ('src/supermimic/data/sampling.py',False),
 ('scripts/data/build_library.py',False),
 ('scripts/data/filter_bones_seed.py',False),
])
def test_source_directories_not_ignored(tmp_path,name,ignored):
 subprocess.run(['git','init','-q',str(tmp_path)],check=True)
 (tmp_path/'.gitignore').write_text((ROOT/'.gitignore').read_text())
 target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True);target.touch()
 result=subprocess.run(['git','check-ignore','--no-index','-q',name],cwd=tmp_path)
 assert result.returncode==(0 if ignored else 1)
