import math
report = {"revision": 1}
strategy = MotionStrategy(ik_solver="pyroki",trajectory_planner="interpolation")
open_gripper()
obs = get_observation("agentview")
frame = get_robot_state().base_frame
goal_text = obs.task_context.language.split("Fixture goal can-center robot0_base XYZ: ")[1]
goal = [float(x) for x in goal_text.split(",")]

def locate():
    fresh = get_observation("agentview")
    mask = []
    for row in fresh.cameras["agentview"].rgb:
        selected = []
        for pixel in row:
            r,g,b = int(pixel[0]),int(pixel[1]),int(pixel[2])
            selected.append(r > 65 and r > g*1.6 and r > b*1.5)
        mask.append(selected)
    points = mask_to_point_cloud(mask,"agentview",target_frame=frame).points
    points = [p for p in points if float(p[1]) < .03 and .25 < float(p[0]) < .95 and -.25 < float(p[2]) < .15]
    if len(points) < 3:
        raise ValueError("not enough can pixels")
    xs = sorted(float(p[0]) for p in points)
    ys = sorted(float(p[1]) for p in points)
    zs = sorted(float(p[2]) for p in points)
    top = [(xs[0]+xs[-1])*.5,(ys[0]+ys[-1])*.5,zs[-1]]
    print("localized can pixels",len(points),"top_center",top)
    return top

can = locate()
quat = get_robot_state().end_effector_poses["primary"].quaternion_wxyz

def move(xyz,label,budget=170):
    outcome = move_to_pose(Pose(position=xyz,quaternion_wxyz=quat,frame=frame),tolerance=.012,max_steps=budget,strategy=strategy)
    print(label,outcome.ok,str(outcome.error),"jaw",float(get_robot_state().gripper_positions["primary"]))
    report[label] = outcome.ok
    return outcome.ok

if move([can[0],can[1],can[2]+.15],"approach"):
    if move([can[0],can[1],can[2]-.040],"grasp_descent"):
        closed = close_gripper()
        print("close",closed.ok,str(closed.error),"jaw",float(get_robot_state().gripper_positions["primary"]))
        transit = max(can[2]+.23,goal[2]+.22)
        if move([can[0],can[1],transit],"lift"):
            start = get_robot_state().end_effector_poses["primary"].position
            carried = True
            for segment in range(1,4):
                if carried:
                    carried = move([float(start[0])+(goal[0]-float(start[0]))*segment/3.,float(start[1])+(goal[1]-float(start[1]))*segment/3.,transit],"carry_"+str(segment))
            if carried:
                if move([goal[0],goal[1],goal[2]+.025],"place"):
                    opened = open_gripper()
                    print("release",opened.ok,str(opened.error))
                    move([goal[0],goal[1],transit],"retreat")
result = report
