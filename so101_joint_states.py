from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower.robot_kinematic_processor import ForwardKinematicsJointsToEE
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import observation_to_transition, transition_to_observation

def get_joint_states(port = "/dev/ttyACM0", robot_id = "ryan_robot"):
    
    follower_config = SO100FollowerConfig(
        port=port,
        id=robot_id,
        use_degrees=False
    )

    follower = SO100Follower(follower_config)
    follower.connect()
    
    urdf_path = "/home/ryan/Documents/so101_ppo/so101_new_calib.urdf"

    kinematics = RobotKinematics(
        urdf_path=str(urdf_path),
        target_frame_name="gripper_frame_link",
        joint_names=list(follower.bus.motors.keys())
    )

    joints_to_ee = RobotProcessorPipeline(
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=kinematics,
                motor_names=list(follower.bus.motors.keys())
            )
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation
    )
               
    obs = follower.get_observation()

    print("=======================")
    print("Joint States:")
    print(obs)
    print()
    print("End-Effector State:")
    print(joints_to_ee(obs))
    print("=======================")

    follower.disconnect()
 

if __name__ == "__main__":
    get_joint_states(
        port = "/dev/ttyACM0",
        robot_id="ryan_robot"
    )