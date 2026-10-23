"""Build floating-base MuJoCo training assets from pinned SOMA-provided URDFs."""
from pathlib import Path
import json, re, xml.etree.ElementTree as ET, ast, math
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def a3_gains():
    params={}
    for node in ast.parse((ROOT/"vendor/a3-sonic/gear_sonic/utils/a3_motor_params.py").read_text()).body:
        if isinstance(node,ast.Assign):params[node.targets[0].id]=eval(compile(ast.Expression(node.value),"motor_parameters","eval"),{"__builtins__":{}},params)
    tree=ast.parse((ROOT/"vendor/a3-sonic/gear_sonic/envs/manager_env/robots/a3.py").read_text())
    out={}
    for call in ast.walk(tree):
        if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!="ImplicitActuatorCfg":continue
        kw={x.arg:x.value for x in call.keywords}
        for key in ["stiffness","damping","effort_limit_sim","armature","velocity_limit_sim"]:
            try:v=eval(compile(ast.Expression(kw[key]),"actuator_parameters","eval"),{"__builtins__":{}},params)
            except (ValueError,KeyError,NameError):continue
            if isinstance(v,dict):out.setdefault(key,{}).update(v)
    return out

def match(table,name,default):
    found=[v for k,v in table.items() if re.fullmatch(k,name)]
    return float(found[-1]) if found else default

for robot,folder,root_name,anchor,suffix in [
    ("x2_ultra","x2","pelvis","torso_link","link"),
    ("a3","a3","pelvis_link","torso_Link","Link")]:
    src=next((ROOT/"vendor/soma-retargeter/soma_retargeter/assets/robotics/agibot"/folder).glob("desc/*.urdf"))
    target=ROOT/"assets/robots"/robot;target.mkdir(parents=True,exist_ok=True)
    tree=ET.parse(src);doc=tree.getroot()
    # Preserve fixed bodies for identical FK indexing between standalone and mjlab models.
    extension=doc.find("mujoco")
    if extension is None:extension=ET.SubElement(doc,"mujoco")
    for old in list(extension):extension.remove(old)
    ET.SubElement(extension,"compiler",fusestatic="false",discardvisual="true",balanceinertia="true")
    limits={}
    for joint in doc.findall("joint"):
        if joint.attrib["type"]!="fixed":
            lim=joint.find("limit");limits[joint.attrib["name"]]=float(lim.get("effort",100)) if lim is not None else 100.
    for mesh in doc.findall(".//mesh"):
        mesh.set("filename",str((src.parent/mesh.attrib["filename"]).resolve()))
    urdf=target/"source.urdf";tree.write(urdf,encoding="unicode")
    spec=mujoco.MjSpec.from_file(str(urdf));spec.body(root_name).add_freejoint(name="floating_base")
    model=spec.compile()
    for i,g in enumerate(spec.geoms):
        body=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[i]))
        foot="ankle_roll" in body
        g.name=("foot_" if foot else "body_")+str(i)+"_collision"
        g.contype=1;g.conaffinity=0;g.condim=3 if foot else 1
        g.friction=[1.,.005,.0001];g.group=0
    spec.option.timestep=.005
    xml=target/"robot.xml";xml.write_text(spec.to_xml());model=mujoco.MjModel.from_xml_path(str(xml))
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(1,model.njnt)]
    assert model.nq==7+len(names) and model.nv==6+len(names)
    gain_table=a3_gains() if robot=="a3" else {}
    xtable={}
    if robot=="x2_ultra":
        xtree=ast.parse((ROOT/"vendor/x2-sonic/scripts/eval_x2_mujoco.py").read_text())
        for node in xtree.body:
            if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ["ARMATURES","EFFORT_LIMITS","DEFAULT_JOINT_POS"]:
                xtable[node.targets[0].id]=ast.literal_eval(node.value)
    home_a3={}
    if robot=="a3":
        atree=ast.parse((ROOT/"vendor/a3-sonic/gear_sonic/envs/manager_env/robots/a3.py").read_text())
        for node in ast.walk(atree):
            if isinstance(node,ast.keyword) and node.arg=="joint_pos":
                home_a3=ast.literal_eval(node.value);break
    kp={};kd={};eff={};home={};armatures={};velocities={}
    for name in names:
        strong=any(x in name for x in ["hip","knee","waist"])
        k=100. if strong else 30.;d=4. if strong else 2.
        if "ankle" in name:k,d=50.,2.
        if "head" in name:k,d=15.,1.
        kp[name]=match(gain_table.get("stiffness",{}),name,k)
        kd[name]=match(gain_table.get("damping",{}),name,d)
        eff[name]=match(gain_table.get("effort_limit_sim",{}),name,limits[name])
        home[name]=home_a3.get(name,0.)
        armatures[name]=match(gain_table.get("armature",{}),name,.01)
        velocities[name]=match(gain_table.get("velocity_limit_sim",{}),name,30.)
        if robot=="x2_ultra":
            arm=next(float(v) for k,v in xtable["ARMATURES"].items() if k in name)
            armatures[name]=arm
            kp[name]=arm*(20*math.pi)**2;kd[name]=4*arm*20*math.pi
            eff[name]=next(float(v) for k,v in xtable["EFFORT_LIMITS"].items() if k in name)
            home[name]=next((float(v) for k,v in xtable["DEFAULT_JOINT_POS"].items() if k in name),0.)
    tracked=[root_name]
    for side in ["left","right"]:
        tracked += [f"{side}_hip_roll_{suffix}",f"{side}_knee_{suffix}",f"{side}_ankle_roll_{suffix}"]
    tracked += [anchor]
    for side in ["left","right"]:
        tracked += [f"{side}_shoulder_roll_{suffix}",f"{side}_elbow_{suffix}",f"{side}_wrist_yaw_{suffix}"]
    for name in tracked:assert mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)>=0,name
    data=mujoco.MjData(model);data.qpos[3:7]=[1,0,0,0];data.qpos[7:]=[home[n] for n in names];mujoco.mj_forward(model,data)
    footids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,f"{side}_ankle_roll_{suffix}") for side in ["left","right"]]
    height=float(-min(data.xpos[i,2] for i in footids)+.08)
    profile={"robot":robot,"num_joints":len(names),"joint_names":names,"xml_path":str(xml),"root_body":root_name,"anchor_body":anchor,"tracked_bodies":tracked,"foot_geom_regex":"^foot_.*_collision$","armature":armatures,"velocity_limit":velocities,"collision_policy":"Robot-ground contacts enabled; self-collisions disabled because raw assembly meshes overlap. Matches official A3 SONIC; documented approximation for X2.","stiffness":kp,"damping":kd,"effort_limit":eff,"default_joint_pos":home,"standing_height":height,"action_scale":{n:.25*eff[n]/kp[n] for n in names},"source_urdf":str(src),"notes":"Rigid sole URDF; serial-joint approximation. X2 training gains from sonic-x2; A3 gains from official SONIC config; both require validation. No hardware validation."}
    (target/"profile.json").write_text(json.dumps(profile,indent=2)+"\n")
    print(robot,model.nq,model.nv,model.nbody,model.ngeom,"height",height,flush=True)
