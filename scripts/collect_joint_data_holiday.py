import argparse
from pathlib import Path
import numpy as np

from robot_payload_id.control import (
    FourierSeriesTrajectory,
)
from robot_payload_id.utils import (
    JointData,
    FourierSeriesTrajectoryAttributes,
)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from hday_motion_planner_msgs.srv import Move

exit_mode = False


class JointStatePublisher(Node):
    def __init__(self):
        super().__init__("joint_state_publisher")

        self.dt = 0.001

        self.publisher_ = self.create_publisher(
            JointState, "/hday/rt_franka/desired_joint_state", 10
        )
        self.timer = self.create_timer(
            self.dt, self.publish_joint_states
        )  # 10Hz 주기로 발행

        self.joint_state_subscriber = self.create_subscription(
            JointState, "/hday/rt_franka/joint_state", self.sub_joint_state, 10
        )

        self.joint_command_subscriber = self.create_subscription(
            JointState,
            "/hday/rt_franka/joint_torque_command",
            self.sub_joint_command,
            10,
        )

        self.motion_planner_client = self.create_client(
            Move, "/hday/motion_planner/move"
        )
        while not self.motion_planner_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Waiting for Motion Planner Server")

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--traj_parameter_path",
            type=Path,
            required=True,
            help="Path to the trajectory parameter folder. The folder must contain "
            + "'a_value.npy', 'b_value.npy', and 'q0_value.npy' or 'control_points.npy', "
            + "'knots.npy', and 'spline_order.npy'.",
        )
        parser.add_argument(
            "--save_data_path",
            type=Path,
            help="Path to save the data to.",
        )
        parser.add_argument(
            "--time_horizon",
            type=float,
            default=10.0,
            help="The time horizon/ duration of the trajectory. Only used for Fourier "
            + "series trajectories.",
        )

        args = parser.parse_args()
        traj_parameter_path = args.traj_parameter_path
        self.save_data_path = args.save_data_path
        self.time_horizon = args.time_horizon

        traj_attrs = FourierSeriesTrajectoryAttributes.load(traj_parameter_path)
        self.excitation_traj = FourierSeriesTrajectory(
            traj_attrs=traj_attrs,
            time_horizon=self.time_horizon,
        )
        self.time_now = 0
        self.ini = True

        self.joint_position = None
        self.joint_velocity = None
        self.joint_torque = None
        self.joint_position_command = None
        self.joint_torque_command = None
        self.joint_positions = []
        self.joint_velocities = []
        self.joint_torques = []
        self.joint_position_commands = []
        self.joint_torque_commands = []
        self.sample_times_s = []

    def sub_joint_state(self, msg: JointState):
        self.joint_position = np.asarray(msg.position, dtype=np.float32)
        self.joint_velocity = np.asarray(msg.velocity, dtype=np.float32)
        self.joint_torque = np.asarray(msg.effort, dtype=np.float32)

    def sub_joint_command(self, msg: JointState):
        self.joint_torque_command = np.asarray(msg.effort, dtype=np.float32)

    def publish_joint_states(self):
        if self.ini:
            joint_state_msg = JointState()
            joint_state_msg.header.stamp = self.get_clock().now().to_msg()
            joint_state_msg.header.frame_id = "map"

            joint_position_command = self.excitation_traj._compute_positions(0)
            joint_state_msg.position = joint_position_command.tolist()
            for i in range(len(joint_state_msg.position)):
                joint_state_msg.name.append("joint" + str(i + 1))

            self.move_srv.stamp = joint_state_msg.header.stamp
            self.move_srv.joint_target = joint_state_msg

            self.future = self.motion_planner_client.call_async(self.move_srv)
            self.move_res_received = False

            if self.future.done():
                self.ini = False

        if not self.ini:
            if self.joint_position is None:
                print("Joint State is not subscribed")
                return

            if self.joint_torque_command is None:
                print("Joint Command is not subscribed")
                return

            joint_position_command = self.excitation_traj._compute_positions(
                self.time_now
            )
            joint_state_msg = JointState()
            joint_state_msg.header.stamp = self.get_clock().now().to_msg()
            for i in range(len(joint_state_msg.position)):
                joint_state_msg.name.append("joint" + str(i + 1))

            joint_state_msg.position = joint_position_command.tolist()
            joint_state_msg.velocity = []
            joint_state_msg.effort = []

            self.publisher_.publish(joint_state_msg)
            # self.get_logger().info(f"Published JointState: {msg.position}")

            self.joint_positions.append(self.joint_position)
            self.joint_velocities.append(self.joint_velocity)
            self.joint_torques.append(self.joint_torque)
            self.joint_position_commands.append(self.joint_position_command)
            self.joint_torque_commands.append(self.joint_torque_command)
            self.sample_times_s.append(self.time_now)

            self.time_now += self.dt
            if self.time_now >= self.time_horizon:

                self.data_save()
                self.destroy_node()
                global exit_mode
                exit_mode = True

    def data_save(self):
        self.joint_positions = np.array(self.joint_positions)
        self.joint_velocities = np.array(self.joint_velocities)
        self.joint_torques = np.array(self.joint_torques)
        self.joint_position_commands = np.array(self.joint_position_commands)
        self.joint_torque_commands = np.array(self.joint_torque_commands)
        self.sample_times_s = np.array(self.sample_times_s)

        joint_data = JointData(
            joint_positions=self.joint_positions,  # (T, N)
            joint_velocities=self.joint_velocities,  # (T, N)
            joint_accelerations=np.zeros_like(self.joint_positions) * np.nan,  # (T, N)
            joint_torques=self.joint_torques,  # (T, N)
            sample_times_s=self.sample_times_s,  # (T,)
        )
        joint_data.save_to_disk(self.save_data_path)

        # Also save commanded data
        np.save(
            self.save_data_path / "commanded_joint_torques.npy",
            self.joint_torque_commands,
        )  # (T, N)
        print(f"Collected {len(self.sample_times_s)} data samples.")


def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()
    try:
        while rclpy.ok() and not exit_mode:
            rclpy.spin_once(node, timeout_sec=0.1)
    except:
        pass
        # traceback.print_exc()
    finally:
        node.destroy_node()
        # rclpy.shutdown()


if __name__ == "__main__":
    main()
