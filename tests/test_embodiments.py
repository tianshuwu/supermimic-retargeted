"""Check custom-embodiment FK/velocities and gradients against MuJoCo on CPU."""
import os,subprocess,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('robot',['a3','x2_ultra'])
def test_embodiment_fk_and_gradients(robot):
 profile=ROOT/f'assets/robots/{robot}/profile.json'
 if not profile.exists():pytest.skip('Generate robot assets first')
 env=os.environ.copy();env['SM_ROBOT']=robot;env['SM_ROBOT_SPEC']=str(profile)
 code=r"""
import mujoco,numpy as np,torch
from supermimic.data.motion_convert import load_g1_model
from supermimic.kinematics.g1_fk import G1Kinematics
m=load_g1_model();assert m.nq==38 and m.nv==37
rng=np.random.default_rng(17);n=8
qp=np.zeros((n,m.nq));qp[:,2]=1.;qp[:,3]=1.
lo=m.jnt_range[1:,0];hi=m.jnt_range[1:,1];qp[:,7:]=rng.uniform(lo,hi,(n,m.nq-7));qv=rng.normal(size=(n,m.nv))
pos=[];vel=[];data=mujoco.MjData(m)
for p,v in zip(qp,qv):
 data.qpos[:]=p;data.qvel[:]=v;mujoco.mj_forward(m,data);pos.append(data.xpos[1:].copy());one=[]
 for body in range(1,m.nbody):
  value=np.zeros(6);mujoco.mj_objectVelocity(m,data,mujoco.mjtObj.mjOBJ_XBODY,body,value,0);one.append(value)
 vel.append(one)
fk=G1Kinematics(m,'cpu',torch.float64);p=torch.tensor(qp,requires_grad=True);v=torch.tensor(qv,requires_grad=True);out=fk(p,v)
np.testing.assert_allclose(out['pos'].detach(),pos,atol=1e-10)
np.testing.assert_allclose(out['lin_vel'].detach(),np.asarray(vel)[:,:,3:],atol=1e-9)
np.testing.assert_allclose(out['ang_vel'].detach(),np.asarray(vel)[:,:,:3],atol=1e-9)
(out['pos'].square().sum()+out['lin_vel'].square().sum()).backward()
assert torch.isfinite(p.grad).all() and torch.isfinite(v.grad).all()
"""
 subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,check=True,timeout=90)
