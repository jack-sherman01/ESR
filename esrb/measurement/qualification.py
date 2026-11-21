"""Analytic physical and computational references with explicit error floors."""
import numpy as np
import mujoco


def qualify():
    rows=[]
    for angle in (0.,.30):
        xml=f'''<mujoco><option timestep="0.002" gravity="0 0 -9.81" tolerance="1e-12" iterations="100"/>
        <worldbody><geom type="plane" size="2 2 .1" euler="0 {angle} 0" friction="2 .01 .001"/>
        <body pos="0 0 .20"><freejoint/><inertial pos=".02 .01 0" mass="2" diaginertia=".02 .02 .02"/>
        <geom type="box" size=".1 .1 .1" friction="2 .01 .001" euler="0 {angle} 0"/></body></worldbody></mujoco>'''
        m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
        # A known force at a lever arm, expressed about the COM.
        force=np.array([.2,0.,0.]);lever=np.array([0.,0.,.05]);torque=np.cross(lever,force)
        d.xfrc_applied[1,:]=np.r_[force,torque]
        for _ in range(3000):mujoco.mj_step(m,d)
        total_force=np.zeros(3);total_moment=np.zeros(3)
        for i,c in enumerate(d.contact):
            wrench=np.zeros(6);mujoco.mj_contactForce(m,d,i,wrench)
            R=c.frame.reshape(3,3);f=R.T@wrench[:3];tau=R.T@wrench[3:]
            sign=1 if m.geom_bodyid[c.geom2]==1 else -1
            total_force+=sign*f
            total_moment+=sign*(tau+np.cross(c.pos-d.xpos[1],f))
        expected_force=np.array([0,0,19.62])-force
        expected_moment=-torque+np.cross(d.xipos[1]-d.xpos[1],expected_force)
        ferr=float(np.linalg.norm(total_force-expected_force));terr=float(np.linalg.norm(total_moment-expected_moment))
        moment_at_com=total_moment-np.cross(d.xipos[1]-d.xpos[1],total_force)
        lever_error=float(np.linalg.norm(moment_at_com+torque))
        rows.append(dict(name=f'static_support_and_lever_rotation_{angle}',expected_force_n=expected_force.tolist(),
            measured_force_n=total_force.tolist(),expected_moment_nm=expected_moment.tolist(),
            measured_moment_nm=total_moment.tolist(),force_abs_error_n=ferr,moment_abs_error_nm=terr,
            expected_com_moment_nm=(-torque).tolist(),measured_com_moment_nm=moment_at_com.tolist(),
            known_lever_abs_error_nm=lever_error,
            force_abs_floor_n=.001,moment_abs_floor_nm=.001,
            status='PASS' if ferr<=max(.001,.05*np.linalg.norm(expected_force)) and terr<=max(.001,.05*np.linalg.norm(expected_moment)) and lever_error<=max(.001,.05*np.linalg.norm(torque)) else 'FAIL'))
    # Frame translation and rotation: two attached points have zero relative motion.
    theta=np.linspace(0.,1.,101);points=[]
    for t in theta:
        R=np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1.]])
        origin=np.array([t*.04,0.,0.]);local=np.array([.12,.03,.01]);world=origin+R@local
        points.append(np.linalg.solve(R,world-origin))
    error=float(np.max(np.abs(np.asarray(points)-points[0])))
    rows.append(dict(name='moving_fixture_coordinates',max_error_m=error,status='PASS' if error<1e-12 else 'FAIL'))
    return dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',checks=rows,
                limitation='single engine analytic fixtures qualify units/transforms, not physical realism')
