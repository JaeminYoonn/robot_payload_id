"""
Similar to `InertiaVisualizer` in drake but allows editing the plant's inertial
parameters.
Visualizes the inertial ellipsoids for a robot arm.
"""

import numpy as np


def main():

    print("===========================================")
    print("============== only_panda ===============")
    print("===========================================")
    data = np.load("joint_data/only_panda.npy", allow_pickle=True).item()
    values = []
    for key, value in data.items():
        if "m" in key and len(key) < 6:
            mass = value
        if "h" in key:
            key_gen = key.replace("h", "p")
            value_gen = value / mass
            values.append(value_gen)
        else:
            values.append(value)

    print(
        len(values)
    )  ## (m, px, py, pz, ixx, ixy, ixz, iyy, iyz, izz, reflected_inertia, viscous_friction, dynamic_dry_friction)

    for i in range(7):
        k = 13 * i
        print("link ", i)
        print(f'<mass value="{values[k]}"/>')
        print(
            f'<origin rpy="{0} {0} {0}" xyz="{values[k+1]} {values[k+2]} {values[k+3]}"/>'
        )
        print(
            f'<inertia ixx="{values[k+4]}" ixy="{values[k+5]}" ixz="{values[k+6]}" iyy="{values[k+7]}" iyz="{values[k+8]}" izz="{values[k+9]}"/>'
        )

    m_pre = values[-13]
    pb_pre = np.array([values[-12], values[-11], values[-10]])
    I_pre = np.array(
        [
            [values[-9], values[-8], values[-7]],
            [values[-8], values[-6], values[-5]],
            [values[-7], values[-5], values[-4]],
        ]
    )

    print("===========================================")
    print("============== panda_and_ft ===============")
    print("===========================================")
    data = np.load("joint_data/panda_and_ft.npy", allow_pickle=True).item()
    values = []
    for key, value in data.items():
        if "m" in key and len(key) < 6:
            mass = value
        if "h" in key:
            key_gen = key.replace("h", "p")
            value_gen = value / mass
            values.append(value_gen)
        else:
            values.append(value)

    m_aft = values[0]
    pb_aft = np.array([values[1], values[2], values[3]])
    I_aft = np.array(
        [
            [values[4], values[5], values[6]],
            [values[5], values[7], values[8]],
            [values[6], values[8], values[9]],
        ]
    )

    ## hard-coding
    d = np.array([0, 0, 0.107])
    m = m_aft - m_pre
    pb = 1 / m * (m_aft * pb_aft - m_pre * pb_pre) - d
    p_tot = pb + d
    I = (
        I_aft
        - I_pre
        - m * (p_tot.T @ p_tot * np.eye(3) - p_tot @ p_tot.T)
        + m * (pb.T @ pb * np.eye(3) - pb @ pb.T)
    )

    print(f'<mass value="{m}"/>')
    print(f'<origin rpy="{0} {0} {0}" xyz="{pb[0]} {pb[1]} {pb[2]}"/>')
    print(
        f'<inertia ixx="{I[0,0]}" ixy="{I[0,1]}" ixz="{I[0,2]}" iyy="{I[1,1]}" iyz="{I[1,2]}" izz="{I[2,2]}"/>'
    )

    print("===========================================")
    print("============== panda_and_ft_and_mount ===============")
    print("===========================================")
    m_pre = m_aft
    pb_pre = pb_aft
    I_pre = I_aft
    d_pre = d

    data = np.load("joint_data/panda_and_ft_and_mount.npy", allow_pickle=True).item()
    values = []
    for key, value in data.items():
        if "m" in key and len(key) < 6:
            mass = value
        if "h" in key:
            key_gen = key.replace("h", "p")
            value_gen = value / mass
            values.append(value_gen)
        else:
            values.append(value)

    m_aft = values[0]
    pb_aft = np.array([values[1], values[2], values[3]])
    I_aft = np.array(
        [
            [values[4], values[5], values[6]],
            [values[5], values[7], values[8]],
            [values[6], values[8], values[9]],
        ]
    )

    ## hard-coding
    d = d_pre + np.array(
        [0.0004147841565186411, -0.00047754315999103223, 0.02 + 0.031223583359233587]
    )
    m = m_aft - m_pre
    pb = 1 / m * (m_aft * pb_aft - m_pre * pb_pre) - d
    p_tot = pb + d
    I = (
        I_aft
        - I_pre
        - m * (p_tot.T @ p_tot * np.eye(3) - p_tot @ p_tot.T)
        + m * (pb.T @ pb * np.eye(3) - pb @ pb.T)
    )

    print(f'<mass value="{m}"/>')
    print(f'<origin rpy="{0} {0} {0}" xyz="{pb[0]} {pb[1]} {pb[2]}"/>')
    print(
        f'<inertia ixx="{I[0,0]}" ixy="{I[0,1]}" ixz="{I[0,2]}" iyy="{I[1,1]}" iyz="{I[1,2]}" izz="{I[2,2]}"/>'
    )

    print("===========================================")
    print("============== panda_and_ft_and_mount_and_gripper ===============")
    print("===========================================")
    m_pre = m_aft
    pb_pre = pb_aft
    I_pre = I_aft
    d_pre = d

    data = np.load(
        "joint_data/panda_and_ft_and_mount_and_gripper.npy", allow_pickle=True
    ).item()
    values = []
    for key, value in data.items():
        if "m" in key and len(key) < 6:
            mass = value
        if "h" in key:
            key_gen = key.replace("h", "p")
            value_gen = value / mass
            values.append(value_gen)
        else:
            values.append(value)

    m_aft = values[0]
    pb_aft = np.array([values[1], values[2], values[3]])
    I_aft = np.array(
        [
            [values[4], values[5], values[6]],
            [values[5], values[7], values[8]],
            [values[6], values[8], values[9]],
        ]
    )

    ## hard-coding
    d = d_pre + np.array([0, 0, 0.025])
    m = m_aft - m_pre
    pb = 1 / m * (m_aft * pb_aft - m_pre * pb_pre) - d
    p_tot = pb + d
    I = (
        I_aft
        - I_pre
        - m * (p_tot.T @ p_tot * np.eye(3) - p_tot @ p_tot.T)
        + m * (pb.T @ pb * np.eye(3) - pb @ pb.T)
    )

    print(f'<mass value="{m}"/>')
    print(f'<origin rpy="{0} {0} {0}" xyz="{pb[0]} {pb[1]} {pb[2]}"/>')
    print(
        f'<inertia ixx="{I[0,0]}" ixy="{I[0,1]}" ixz="{I[0,2]}" iyy="{I[1,1]}" iyz="{I[1,2]}" izz="{I[2,2]}"/>'
    )


if __name__ == "__main__":
    main()
