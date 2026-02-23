import argparse
import sys
import time
import numpy as np
import torch
import xml.etree.ElementTree as ET
from pathlib import Path

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.processor import RobotAction
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import observation_to_transition, transition_to_observation
from lerobot.robots.so_follower.robot_kinematic_processor import ForwardKinematicsJointsToEE

sys.path.insert(0, "/home/ryan/Documents/IsaacLab/scripts/ryan_ppo")
from network import Actor

class deploy_reach:

    def __init__(
        self,
        agent_path,
        urdf_path,
        port,
        action_scale,
        checkpoint_path = "so101_ppo/reach_agent.pth",
        target_pose = np.array([0.25, 0.1, 0.25, 0.0, 0.0, 1.0, 0.0]),
        device = "cpu",
        robot_id = "ryan_robot",
        use_normalization = True,
        control_hz = 15.0
    ):
        """Initialize class with robot interfaces, agent network, and robot kinematics"""

        self.urdf_path = urdf_path
        self.port = port
        self.target_pose = target_pose
        self.action_scale = action_scale
        self.device = torch.device(device)
        self.control_dt = 1.0/control_hz

        follower_config = SO100FollowerConfig(
            port=port,
            id=robot_id,
            use_degrees=False,
        )
        self.follower = SO100Follower(follower_config)
        self.follower.connect()

        # [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
        self.default_joint_state = np.array([
            0.0,
            0.0,
            0.0,
            94.69,
            -1.75,
        ], dtype=np.float32)

        self.default_gripper_state = 9.09

        self.joint_limits = self.parse_joint_limits_from_urdf(urdf_path)

        self.actor = Actor(
            state_dim=24,
            action_dim=5,
            hidden_dims=[256,128,64],
            use_normalization=use_normalization,
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint)
        self.actor.eval()

        self.motor_names = list(self.follower.bus.motors.keys())
        self.num_joints = 6
        self.action_dim = 5

    def move_to_start(self, duration = 3.0):
        """Moves the robot arm into the start position."""
        current_obs = self.follower.get_observation()
        gripper_name = self.motor_names[self.action_dim]
        current_arm = np.array([current_obs[f"{m}.pos"] for m in self.motor_names[:self.action_dim]])
        current_gripper = float(current_obs[f"{gripper_name}.pos"])

        steps = 3 * 15
        step_size = 1.0/15.0
        for step in range(steps):
            alpha = (step + 1) / steps
            target_arm = current_arm * (1 - alpha) + self.default_joint_state * alpha
            target_gripper = current_gripper * (1 - alpha) + self.default_gripper_state * alpha
            action_dict = {f"{m}.pos": float(target_arm[i]) for i, m in enumerate(self.motor_names[:self.action_dim])}
            action_dict[f"{gripper_name}.pos"] = float(target_gripper)
            robot_action = RobotAction(action_dict)
            self.follower.send_action(robot_action)
            time.sleep(step_size)

        print("Robot Reached Start Position")

    def get_observation(self):
        """Constructs the observation array to be used as input to the actor network."""
        obs_dict = self.follower.get_observation()
        gripper_name = self.motor_names[self.action_dim]
        arm_state = np.array(
            [obs_dict[f"{m}.pos"] for m in self.motor_names[:self.action_dim]]
        )
        gripper_state = float(obs_dict[f"{gripper_name}.pos"])

        # joint state relative to starting position. IsaacLab uses this format
        arm_pos_rel = arm_state - self.default_joint_state  # 5D

        gripper_state = np.array([2.0 * (gripper_state - self.default_gripper_state)])

        joint_state = np.concatenate([arm_pos_rel, gripper_state])  # 6D

        # joint velocities
        current_time = time.time()
        arm_rad = self.arm_state_to_radians(arm_state)
        gripper_rad = self.gripper_to_radians(gripper_state)
        all_rad = np.append(arm_rad, gripper_rad).astype(np.float32)  # 6D

        if self.prev_joint_state_rad is not None and self.prev_time is not None:
            dt = current_time - self.prev_time
            joint_vel = (all_rad - self.prev_joint_state_rad) / dt if dt > 0 else np.zeros(self.num_joints, dtype=np.float32)
        else:
            joint_vel = np.zeros(self.num_joints, dtype=np.float32)

        self.prev_joint_state_rad = all_rad
        self.prev_time = current_time

        # observation format: [joint_state, joint_vel, target_pose, last_action]
        # joint_state = 6-D, all joint motor values in [-100,100]. [0, 100] for gripper.
        # joint_vel = 6-D, all joint velocities in rad/s
        # target_pose = 7-D, XYZ and quaternion
        # last_action = 5-D, positions for 5 arm motors commanded at last timestep.
        observation = np.concatenate([
            joint_state,  # 6D: relative positions
            joint_vel,           # 6D: velocities in rad/s
            self.target_pose,    # 7D: target pose
            self.last_action    # 5D: previous action
        ])

        return observation
    
    def parse_joint_limits_from_urdf(self, urdf_path):
        """Finds joint limits for robot from URDF file.
           Necessary for converting the normalized state output by LeRobot to radians for joint velocities."""
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        limits = {}
        for joint in root.findall('joint'):
            joint_name = joint.get('name')
            joint_type = joint.get('type')
            
            if joint_type == 'revolute':
                limit_elem = joint.find('limit')
                if limit_elem is not None:
                    lower = float(limit_elem.get('lower'))
                    upper = float(limit_elem.get('upper'))
                    limits[joint_name] = (lower, upper)
        
        return limits
    
    def arm_state_to_radians(self, arm_joint_state):
        """Converts normalized arm joint state ouptut by LeRobot to radians for velocity calculation."""
        radians = np.zeros_like(arm_joint_state)

        for i, motor_name in enumerate(self.motor_names[:self.action_dim]):
            lower, upper = self.joint_limits[motor_name]
            radians[i] = (arm_joint_state[i] + 100.0) / 200.0 * (upper - lower) + lower
        return radians
    
    def gripper_to_radians(self, gripper_state):
        """Converts normalized gripper joint state ouptut by LeRobot to radians for velocity calculation."""
        g_lo, g_hi = self.joint_limits["gripper"]

        return (gripper_state / 100.0) * (g_hi-g_lo) + g_lo
    
    def step(self, observation):
        """Retrieves observation from robot, feeds that into the actor network, determines the next action."""
        obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)

        with torch.no_grad():
            mu, std = self.actor(obs_tensor)
            action = mu.squeeze(0).cpu().numpy()
            action_std = std.squeeze(0).cpu().numpy()

        self.last_action = action.copy()

        scaled_action = action * self.action_scale

        target_positions = scaled_action + self.default_joint_state

        action_dict = {}
        for i, motor_name in enumerate(self.motor_names[:self.action_dim]):
            target_clamped = np.clip(target_positions[i], -100.0, 100.0)
            action_dict[f"{motor_name}.pos"] = float(target_clamped)
        
        robot_action = RobotAction(action_dict)
        self.follower.send_action(robot_action)

        return action
    
    def run_episode(self, max_steps=100, reset_to_home=True):
        """Handles the episode logic, retrieving observation, performing the action on the real-robot, and waiting until the next time to perform the loop."""
        
        if reset_to_home:
            self.move_to_start(3.0)

        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self.prev_joint_state_rad = None
        self.prev_time = None

        for step in range(max_steps):
            step_start = time.time()

            obs = self.get_observation()
            action = self.step(obs)

            sleep_time = max(0, self.control_dt)

            if sleep_time:
                time.sleep(sleep_time)
    
    def disconnect(self):
        """Handles"""
        self.follower.disconnect()

def main():
    parser = argparse.ArgumentParser(description="Deploy PPO policy to SO-101")
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to agent (.pth)")
    parser.add_argument("--use-normalization", action="store_true", default=True,
                        help="Use observation normalization (default: True)")

    # Target pose
    parser.add_argument("--target-x", type=float, default=0.25)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--target-z", type=float, default=0.15)
    parser.add_argument("--target-quat", type=float, nargs=4, 
                        default=[0.0, 0.0, 1.0, 0.0],
                        help="Target quaternion [qw, qx, qy, qz]")
    
    # Control
    parser.add_argument("--hz", type=float, default=2.0,
                        help="Control frequency (Hz)")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum steps per episode")
    parser.add_argument("--reset-to-home", action="store_true",
                        help="Reset to home position before episode")
    parser.add_argument("--action-scale", type=float, default=10.0,
                        help="Action scale in normalized space (default: 10.0, training used 30.0)")

    # Robot
    parser.add_argument("--port", type=str, default="/dev/ttyACM0",
                        help="Robot USB port")
    parser.add_argument("--robot-id", type=str, default="ryan_robot",
                        help="Robot calibration ID")
    
    args = parser.parse_args()
    
    # Create target pose
    target_pose = np.array([
        args.target_x, args.target_y, args.target_z,
        *args.target_quat
    ], dtype=np.float32)
    
    # Create deployment
    deployment = deploy_reach(
        agent_path=args.checkpoint,
        urdf_path="/home/ryan/Documents/so101_ppo/so101_new_calib.urdf",
        port=args.port,
        robot_id=args.robot_id,
        target_pose=target_pose,
        action_scale=args.action_scale,
        control_hz=args.hz,
        use_normalization=args.use_normalization
    )
    
    deployment.run_episode(
        max_steps=args.max_steps,
        reset_to_home=args.reset_to_home,
    )
    
    deployment.disconnect()

if __name__=="__main__":
    main()