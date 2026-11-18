import math

report = {"revision": 1, "perception": "public RGB-D red/green masks"}
strategy = MotionStrategy(ik_solver="pyroki", trajectory_planner="interpolation")
open_gripper()
obs = get_observation("agentview")
pixels = obs.cameras["agentview"].rgb
frame = get_robot_state().base_frame

def locate(color):
    mask = []
    for row in pixels:
        selected = []
        for pixel in row:
            r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
            if color == "red":
                selected.append(r > 55 and r > g*1.5 and r > b*1.4)
            else:
                selected.append(g > 45 and g > r*1.3 and g > b*1.2)
        mask.append(selected)
    points = mask_to_point_cloud(mask,"agentview",target_frame=frame).points
    if len(points) < 6:
        raise ValueError("not enough public RGB-D pixels for "+color)
    xs = sorted(float(p[0]) for p in points)
    ys = sorted(float(p[1]) for p in points)
    zs = sorted(float(p[2]) for p in points)
    center = [(xs[0]+xs[-1])*.5,(ys[0]+ys[-1])*.5,zs[-1]]
    print("localized",color,"pixels",len(points),"top_center",center)
    return center

red = locate("red")
green = locate("green")
quat = get_robot_state().end_effector_poses["primary"].quaternion_wxyz

def move(xyz,label,budget=190):
    outcome = move_to_pose(Pose(position=xyz,quaternion_wxyz=quat,frame=frame),
                           tolerance=.012,max_steps=budget,strategy=strategy)
    print(label,outcome.ok,str(outcome.error),"jaw",float(get_robot_state().gripper_positions["primary"]))
    report[label] = outcome.ok
    return outcome.ok

if move([red[0],red[1],red[2]+.12],"approach"):
    if move([red[0],red[1],red[2]-.018],"grasp_descent"):
        closed = close_gripper()
        print("close",closed.ok,str(closed.error),"jaw",float(get_robot_state().gripper_positions["primary"]))
        if move([red[0],red[1],red[2]+.16],"lift"):
            transit = max(red[2],green[2])+.16
            if move([green[0],green[1],transit],"carry"):
                if move([green[0],green[1],green[2]+.025],"place"):
                    opened = open_gripper()
                    print("release",opened.ok,str(opened.error))
                    move([green[0],green[1],transit],"retreat")
result = report
