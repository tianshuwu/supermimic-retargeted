"""Convert references with auditable contact correction (strict LAFAN1 by default)."""
import argparse,os,json,csv
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--robot",required=True);p.add_argument("--workers",type=int,default=8);p.add_argument("--source",type=Path);p.add_argument("--dest",type=Path);p.add_argument("--manifest",type=Path);p.add_argument("--input-fps",type=float,default=30.);p.add_argument("--audit",type=Path);p.add_argument("--skip-invalid",action="store_true");p.add_argument("--include-lafan",action="store_true");args=p.parse_args()
ROOT=Path(__file__).resolve().parents[1];os.environ["SM_ROBOT"]=args.robot
if args.robot!="g1":os.environ["SM_ROBOT_SPEC"]=str(ROOT/"assets/robots"/args.robot/"profile.json")
os.environ["OMP_NUM_THREADS"]="1"
import multiprocessing as mp
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from supermimic.data.motion_convert import load_g1_model,load_csv,resample_motion,forward_kinematics,G1_JOINT_NAMES
MODEL=None

def init():
 global MODEL
 MODEL=load_g1_model()

def convert(job):
 src,dst=map(Path,job);raw=load_csv(src);motion=resample_motion(raw,args.input_fps,50.);offset=0.
 if args.robot!="g1":
  data=mujoco.MjData(MODEL);foot=[]
  for g in range(MODEL.ngeom):
   if not MODEL.geom(g).name.startswith("foot_"):continue
   mid=MODEL.geom_dataid[g];vertices=MODEL.mesh_vert[MODEL.mesh_vertadr[mid]:MODEL.mesh_vertadr[mid]+MODEL.mesh_vertnum[mid]];foot.append((g,vertices[ConvexHull(vertices).vertices].copy()))
  if not foot:raise ValueError("No foot collision geometry")
  bottoms=[];upright=[];heights=[]
  for i in range(0,motion.num_frames,max(5,motion.num_frames//2000)):
   data.qpos[:3]=motion.root_pos[i];data.qpos[3:7]=motion.root_quat[i];data.qpos[7:]=motion.joint_pos[i];mujoco.mj_kinematics(MODEL,data)
   bottom=min(float((v@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g])[:,2].min()) for g,v in foot)
   bottoms.append(bottom);upright.append(Rotation.from_quat(motion.root_quat[i][[1,2,3,0]]).as_matrix()[2,2]);heights.append(motion.root_pos[i,2])
  stand=(np.array(upright)>.85)&(np.array(heights)>.7*np.quantile(heights,.75))
  calibration=np.array(bottoms)[stand] if np.any(stand) else np.array(bottoms)
  offset=-float(np.quantile(calibration,.1))
  if abs(offset)>.25:raise ValueError(f"Suspicious height correction {offset}m in {src}")
  motion.root_pos[:,2]+=offset
  # Repair penetration by changing bounded leg joints, rather than lifting a whole
  # fall/get-up clip and making its standing sections float above the ground.
  leg_ids=np.array([i for i,n in enumerate(G1_JOINT_NAMES) if any(t in n for t in ["hip","knee","ankle"])])
  lo=MODEL.jnt_range[1:,0][leg_ids]+1e-6;hi=MODEL.jnt_range[1:,1][leg_ids]-1e-6
  repaired=0;max_change=0.;previous=None;residual_min=0.;lifts=np.zeros(motion.num_frames)
  for i in range(motion.num_frames):
   data.qpos[:3]=motion.root_pos[i];data.qpos[3:7]=motion.root_quat[i];data.qpos[7:]=motion.joint_pos[i]
   base=data.qpos.copy();original=motion.joint_pos[i,leg_ids].copy()
   def foot_height(x):
    data.qpos[:]=base;data.qpos[7+leg_ids]=x;mujoco.mj_kinematics(MODEL,data)
    return np.array([float((v@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g])[:,2].min()) for g,v in foot])
   before=foot_height(original)
   if before.min()<-.008:
    prior=original if previous is None else previous
    def residual(x):return np.r_[(x-original)*.05,(x-prior)*.02,np.maximum(.003-foot_height(x),0)*20.]
    opt=least_squares(residual,np.clip(prior,lo,hi),bounds=(lo,hi),max_nfev=24,ftol=1e-4,xtol=1e-4,gtol=1e-4)
    after=foot_height(opt.x)
    if after.min()<-.02:
     opt=least_squares(residual,opt.x,bounds=(lo,hi),max_nfev=80,ftol=1e-5,xtol=1e-5,gtol=1e-5)
     after=foot_height(opt.x)
    if after.min()<-.02:
     # Falls can place a leg at a limit-bound local minimum. Try distinct bounded
     # seeds before resorting to the small residual root-height correction.
     for seed in [original,np.zeros(len(leg_ids)),(lo+hi)/2]:
      candidate=least_squares(residual,np.clip(seed,lo,hi),bounds=(lo,hi),max_nfev=200,ftol=1e-6,xtol=1e-6,gtol=1e-6)
      candidate_height=foot_height(candidate.x)
      if candidate_height.min()>after.min():opt,after=candidate,candidate_height
      if after.min()>=-.008:break
    lifts[i]=max(0.,.003-float(after.min()))
    if lifts[i]>.06:raise ValueError(f"Unrepaired foot penetration {after.min()}m in {src} frame {i}")
    motion.joint_pos[i,leg_ids]=opt.x;repaired+=1;max_change=max(max_change,float(abs(opt.x-original).max()));residual_min=min(residual_min,float(after.min()))
   previous=motion.joint_pos[i,leg_ids].copy()
  # A bounded, rate-limited vertical envelope handles residual contact error.
  # At most 6 cm, at most 0.2 m/s, leaving airborne intervals intact.
  for k in range(1,len(lifts)):lifts[k]=max(lifts[k],lifts[k-1]-.004)
  for k in range(len(lifts)-2,-1,-1):lifts[k]=max(lifts[k],lifts[k+1]-.004)
  motion.root_pos[:,2]+=lifts
  motion.root_lin_vel_w=np.gradient(motion.root_pos,.02,axis=0)
  motion.joint_vel=np.gradient(motion.joint_pos,.02,axis=0)

 bodies=forward_kinematics(MODEL,motion)
 arrays=dict(fps=np.array([50.],dtype=np.float32),joint_pos=motion.joint_pos.astype(np.float32),joint_vel=motion.joint_vel.astype(np.float32),body_pos_w=bodies.pos_w.astype(np.float32),body_quat_w=bodies.quat_w.astype(np.float32),body_lin_vel_w=bodies.lin_vel_w.astype(np.float32),body_ang_vel_w=bodies.ang_vel_w.astype(np.float32),joint_names=np.array(G1_JOINT_NAMES),body_names=np.array(bodies.body_names),source=np.array(str(src)),input_fps=np.array([args.input_fps],dtype=np.float32))
 for key,value in arrays.items():
  if np.issubdtype(value.dtype,np.number) and not np.isfinite(value).all():raise ValueError(f"Nonfinite {key}: {src}")
 lo=MODEL.jnt_range[1:,0];hi=MODEL.jnt_range[1:,1]
 violation=float(np.maximum(lo-motion.joint_pos,motion.joint_pos-hi).clip(0).max())
 if args.robot!="g1" and violation>.05:raise ValueError(f"Joint limit violation {violation}: {src}")
 dst.parent.mkdir(parents=True,exist_ok=True);np.savez(dst,**arrays)
 return dict(max_residual_root_lift_m=float(lifts.max()) if args.robot!="g1" else 0.,repaired_frames=repaired if args.robot!="g1" else 0,max_joint_repair_rad=max_change if args.robot!="g1" else 0.,minimum_repaired_sole_m=residual_min if args.robot!="g1" else 0.,clip=src.stem,frames=motion.num_frames,duration_s=(motion.num_frames-1)/50.,root_height_offset_m=offset,joint_limit_violation_rad=violation,joint_velocity_p99=float(np.quantile(abs(motion.joint_vel),.99)))

def guarded_convert(job):
 try:return convert(job)
 except Exception as exc:
  if not args.skip_invalid:raise
  return {"clip":Path(job[0]).stem,"rejected":type(exc).__name__+": "+str(exc)}

if __name__=="__main__":
 if args.manifest:
  with args.manifest.open() as f:selected=[r["filename"] for r in csv.DictReader(f)]
 else:
  with (ROOT/"manifests/lafan1.csv").open() as f:selected=[r["filename"] for r in csv.DictReader(f)]
 source=args.source or ROOT/"data"/("lafan1-retargeted/g1" if args.robot=="g1" else "lafan1-retarget-"+args.robot)
 dest=args.dest or ROOT/"data/npz"/args.robot/"lafan1"
 lookup={p.stem:p for p in source.rglob("*.csv")};missing=set(selected)-lookup.keys();assert not missing,missing
 jobs=[(str(lookup[n]),str(dest/(n+".npz"))) for n in selected]
 with mp.Pool(args.workers,initializer=init) as pool:report=list(pool.imap(guarded_convert,jobs))
 # Failed re-runs must not leave previously accepted files in the training set.
 for row in report:
  if "rejected" in row:(dest/(row["clip"]+".npz")).unlink(missing_ok=True)
 accepted=[r for r in report if "rejected" not in r]
 dest.mkdir(parents=True,exist_ok=True)
 with (dest/"manifest_filtered.csv").open("w",newline="") as f:
  writer=csv.writer(f);writer.writerow(["name","frames","keep","reason"])
  for row in report:writer.writerow([row["clip"],row.get("frames",0),int("rejected" not in row),row.get("rejected","")])
 if args.include_lafan:
  for src in (ROOT/"data/npz"/args.robot/"lafan1").glob("*.npz"):
   target=dest/src.name
   if not target.exists():target.symlink_to(src.resolve())
 out=args.audit or ROOT/"logs/overnight"/(args.robot+"-lafan-preparation.json");out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps({"robot":args.robot,"clips":len(accepted),"rejected":len(report)-len(accepted),"hours":sum(x["duration_s"] for x in accepted)/3600.,"audit":str(out)},indent=2))
