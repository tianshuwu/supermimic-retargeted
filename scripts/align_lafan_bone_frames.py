"""Align LAFAN limb frames and the anatomical pelvis pivot for SOMA."""
import argparse,json,warnings
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from soma_retargeter.io.bvh import BVHImporter
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument("--pelvis-pivot",action="store_true");args=p.parse_args()
flip_left=R.from_rotvec([np.pi,0,0]);flip_right=R.from_rotvec([0,np.pi,0]);identity=R.identity();reports=[]
for src in sorted(args.input.glob('*.bvh')):
 root=BVHImporter.bvh_parser(str(src));nodes=[]
 def walk(node,parent=-1):
  i=len(nodes);nodes.append((node,parent))
  for child in node.children:walk(child,i)
 walk(root);pivot=np.zeros(3)
 if args.pelvis_pivot:
  canonical=BVHImporter.bvh_parser(str(Path(__file__).resolve().parents[1]/'vendor/soma-retargeter/soma_retargeter/assets/soma/soma_zero_frame0.bvh'));cn={}
  def collect(n):
   cn[n.name]=n
   for child in n.children:collect(child)
  collect(canonical);sn={node.name:node for node,_ in nodes}
  pivot=np.mean([sn[n].offset[:3] for n in ['LeftLeg','RightLeg']],axis=0)-np.mean([cn[n].offset[:3] for n in ['LeftLeg','RightLeg']],axis=0)
 corrections=[flip_left if node.name.startswith('Left') else flip_right if node.name.startswith('Right') else identity for node,_ in nodes]
 data=np.concatenate([np.asarray(node.animation) for node,_ in nodes],axis=1);out=data.copy();start=0;old_positions=[];new_positions=[];old_rotations=[];new_rotations=[];offsets={};translation_error=0.
 for i,(node,parent) in enumerate(nodes):
  cp=corrections[parent] if parent>=0 else identity;c=corrections[i];values=np.asarray(node.animation);rot_ids=[k for k,n in enumerate(node.channels) if 'rotation' in n];pos_ids=[k for k,n in enumerate(node.channels) if 'position' in n];order=''.join(node.channels[k][0] for k in rot_ids)
  rotation=R.from_euler(order,values[:,rot_ids],degrees=True);converted=cp.inv()*rotation*c
  with warnings.catch_warnings():
   warnings.simplefilter('ignore');out[:,start+np.array(rot_ids)]=converted.as_euler(order,degrees=True)
  position=values[:,pos_ids] if pos_ids else np.broadcast_to(np.array(node.offset[:3]),(len(values),3)).copy()
  original_position=position.copy()
  if args.pelvis_pivot:
   if parent<0:position=position+rotation.apply(np.broadcast_to(pivot,position.shape).copy())
   elif parent==0:position=position-pivot
  converted_position=cp.inv().apply(position)
  if pos_ids:out[:,start+np.array(pos_ids)]=converted_position
  offset=np.array(node.offset[:3],dtype=float)
  if args.pelvis_pivot:
   if parent<0:offset=offset+rotation[0].apply(pivot)
   elif parent==0:offset=offset-pivot
  offsets[node.name]=cp.inv().apply(offset);start+=len(node.channels)
  old_r=rotation.as_matrix();new_r=converted.as_matrix()
  if parent>=0:
   old_positions.append(old_positions[parent]+np.einsum('tij,tj->ti',old_rotations[parent],original_position));new_positions.append(new_positions[parent]+np.einsum('tij,tj->ti',new_rotations[parent],converted_position));old_rotations.append(old_rotations[parent]@old_r);new_rotations.append(new_rotations[parent]@new_r)
  else:old_positions.append(original_position.copy());new_positions.append(converted_position.copy());old_rotations.append(old_r);new_rotations.append(new_r)
  if not args.pelvis_pivot or i>0:translation_error=max(translation_error,float(abs(old_positions[i]-new_positions[i]).max()))
 assert translation_error<1e-7,translation_error
 text=src.read_text();header=text.split('MOTION')[0];lines=[];current=None;end_site=False
 for line in header.splitlines():
  tokens=line.split()
  if tokens and tokens[0] in ('ROOT','JOINT'):current=tokens[1]
  if tokens[:2]==['End','Site']:end_site=True
  if tokens and tokens[0]=='OFFSET':
   if end_site:
    j=next(i for i,(node,_) in enumerate(nodes) if node.name==current);value=corrections[j].inv().apply(list(map(float,tokens[1:4])));end_site=False
   else:value=offsets[current]
   line=line[:len(line)-len(line.lstrip())]+'OFFSET '+' '.join(f'{v:.9f}' for v in value)
  lines.append(line)
 args.output.mkdir(parents=True,exist_ok=True);dest=args.output/src.name
 with dest.open('w') as f:
  f.write('\n'.join(lines)+'\nMOTION\nFrames: '+str(len(out))+'\nFrame Time: '+str(root.frame_time)+'\n');np.savetxt(f,out,fmt='%.8f')
 reports.append({'clip':src.stem,'frames':len(out),'world_joint_position_max_error_cm':translation_error,'pelvis_pivot_local_cm':pivot.tolist(),'position_invariance_excludes_root_hips':args.pelvis_pivot,'correction':'Left limbs Rx(pi), right limbs Ry(pi); matching local translation/frame gauge transformation; torso/root unchanged'})
(args.output/'alignment.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps({'clips':len(reports),'max_world_position_error_cm':max(x['world_joint_position_max_error_cm'] for x in reports)},indent=2))
