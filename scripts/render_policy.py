"""Render the physical policy and a translucent kinematic reference in one scene."""
import argparse,os,subprocess,json,datetime,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--robot",required=True);p.add_argument("--ckpt",type=Path,required=True);p.add_argument("--clip",default="walk1_subject1");p.add_argument("--seconds",type=float,default=20.);p.add_argument("--record-start-seconds",type=float,default=0.);p.add_argument("--out",type=Path,required=True);p.add_argument("--npz-dir",type=Path);args=p.parse_args()
ROOT=Path(__file__).resolve().parents[1];os.environ["SM_ROBOT"]=args.robot;os.environ["SM_ROBOT_SPEC"]=str(ROOT/"assets/robots"/args.robot/"profile.json");os.environ["MUJOCO_GL"]="egl"
import mujoco,numpy as np,torch
from supermimic.envs import make_env,SuperMimicEnvCfgArgs
from supermimic.eval.benchmark import load_policy_from_checkpoint
checkpoint_stat=args.ckpt.stat()
with args.ckpt.open("rb") as checkpoint_file:
 checkpoint_sha256=hashlib.file_digest(checkpoint_file,"sha256").hexdigest()
policy,cfg=load_policy_from_checkpoint(args.ckpt,"cuda:0")
loaded_stat=args.ckpt.stat()
assert all(getattr(loaded_stat,k)==getattr(checkpoint_stat,k) for k in ("st_dev","st_ino","st_size","st_mtime_ns")),"Checkpoint changed while loading; use an immutable checkpoint."
dataset=args.npz_dir or ROOT/"data/npz"/args.robot/"lafan1"
files=[str(dataset/(args.clip+".npz"))]
env=make_env(SuperMimicEnvCfgArgs(npz_files=files,num_envs=1,action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,sampling_mode="start",record_transitions=False,episode_length_s=1e5,anchor_obs=cfg["env"].get("anchor_obs",False)),"cuda:0")
cmd=env.command_manager.get_term("motion");cmd.forced_frame[:]=cmd.library.clip_start[0];obs,_=env.reset()
model=env.sim.mj_model;physical=mujoco.MjData(model);reference=mujoco.MjData(model);renderer=mujoco.Renderer(model,height=720,width=1280)
ghost_scene=mujoco.MjvScene(model,maxgeom=10000);ghost_option=mujoco.MjvOption()
robot_root_ids=set(int(b) for b in model.jnt_bodyid[np.asarray(model.jnt_type)==mujoco.mjtJoint.mjJNT_FREE])
assert len(robot_root_ids)==1,"Expected one floating-base physical robot"
ghost_rgba=np.array([0.1,0.8,1.0,0.32],dtype=np.float32)
cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.distance=3.2 if args.robot!="x2_ultra" else 2.8;cam.azimuth=135;cam.elevation=-15
args.out.parent.mkdir(parents=True,exist_ok=True)
writer=subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-vcodec","rawvideo","-pix_fmt","rgb24","-s","1280x720","-r","25","-i","-","-an","-vf","drawtext=text='Solid = policy (physics)   Cyan ghost = reference':x=18:y=18:fontcolor=white:fontsize=24","-c:v","libx264","-preset","fast","-crf","23",str(args.out)],stdin=subprocess.PIPE)
assert args.record_start_seconds>=0 and args.seconds>0
record_start_step=round(args.record_start_seconds*50)
assert record_start_step<cmd.library.num_frames-1,"Recording start exceeds clip length"
frames_written=0;rewards=[];stopped=False;steps_completed=0
try:
 with torch.no_grad():
  for step in range(min(record_start_step+int(args.seconds*50),cmd.library.num_frames-1)):
   action=policy(obs["policy"],obs["goal"])
   obs,reward,term,tout,extra=env.step(action)
   steps_completed=step+1
   if step>=record_start_step:rewards.append(float(reward[0]))
   if step>=record_start_step and (step-record_start_step)%2==0:
    physical.qpos[:]=env.sim.data.qpos[:].cpu().numpy()[0];physical.qvel[:]=env.sim.data.qvel[:].cpu().numpy()[0]
    qp,qv=cmd.library.ref_qpos_qvel(cmd.frame);reference.qpos[:]=qp.cpu().numpy()[0];reference.qpos[:3]+=env.scene.env_origins[0].cpu().numpy();reference.qvel[:]=qv.cpu().numpy()[0]
    cam.lookat[:]=(reference.qpos[:3]+physical.qpos[:3])*.5+np.array([0,0,.15])
    mujoco.mj_forward(model,reference);mujoco.mj_forward(model,physical)
    renderer.update_scene(physical,camera=cam)
    mujoco.mjv_updateScene(model,reference,ghost_option,None,cam,mujoco.mjtCatBit.mjCAT_ALL,ghost_scene)
    scene=renderer.scene
    for i in range(ghost_scene.ngeom):
     src=ghost_scene.geoms[i]
     if src.objtype!=mujoco.mjtObj.mjOBJ_GEOM or src.objid<0 or int(model.body_rootid[model.geom_bodyid[src.objid]]) not in robot_root_ids:continue
     if scene.ngeom>=scene.maxgeom:raise RuntimeError("Ghost scene geometry capacity exceeded")
     dst=scene.geoms[scene.ngeom]
     mujoco.mjv_initGeom(dst,src.type,src.size,src.pos,src.mat.ravel(),ghost_rgba)
     dst.dataid=src.dataid;dst.modelrbound=src.modelrbound
     dst.category=mujoco.mjtCatBit.mjCAT_DECOR;dst.objtype=mujoco.mjtObj.mjOBJ_UNKNOWN;dst.objid=-1
     dst.matid=-1;dst.texid=-1;dst.transparent=1;dst.emission=.15
     scene.ngeom+=1
    writer.stdin.write(renderer.render().tobytes());frames_written+=1
   if bool(term[0]) or bool(tout[0]):
    stopped=True;print("Stopped at",step,"terminated",bool(term[0]),flush=True);break
finally:
 writer.stdin.close();code=writer.wait();renderer.close();env.close()
 if code:raise SystemExit(code)
args.out.with_suffix(".json").write_text(json.dumps({"robot":args.robot,"checkpoint":str(args.ckpt.resolve()),"checkpoint_sha256":checkpoint_sha256,"clip":args.clip,"dataset":str(dataset.resolve()),"rendered_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"requested_seconds":args.seconds,"simulated_seconds":max(0,steps_completed-record_start_step)*.02,"record_start_seconds":args.record_start_seconds,"warmup_simulated_seconds":min(steps_completed,record_start_step)*.02,"video_frames":frames_written,"fps":25,"terminated_or_truncated":stopped,"mean_step_reward":sum(rewards)/max(1,len(rewards)),"render_mode":"reference_ghost_overlay","reference_rgba":ghost_rgba.tolist(),"reference_alignment":"Original world coordinates plus environment origin; no per-frame pose alignment or translation offset.","note":"Solid robot: closed-loop physical policy. Cyan translucent robot: kinematic reference, visual only and absent from physics contacts. One shared scene and camera. This video covers one excerpt, not the full benchmark."},indent=2)+"\n")
print(args.out,flush=True)
