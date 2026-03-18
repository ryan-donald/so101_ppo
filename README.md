[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=fff)](https://docs.python.org/3/whatsnew/3.10.html)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-ee4c2c?logo=pytorch&logoColor=white)](https://github.com/pytorch/pytorch/releases/tag/v2.7.0)
[![Hugging Face](https://img.shields.io/badge/LeRobot-FFD21E?logo=huggingface&logoColor=000)](https://github.com/huggingface/lerobot)
[![SO-ARM101](https://img.shields.io/badge/Robot-SO-ARM101-blue)](https://github.com/TheRobotStudio/SO-ARM100)

This is a repository containing scripts to deploy a trained PPO agent on a real-world SO-ARM101 Robot. 
Information about the SO-ARM101 robot can be found [here](https://github.com/TheRobotStudio/SO-ARM100).

The agent is trained using my PPO implementation [here](https://github.com/ryan-donald/PPO_IsaacLab). An environment mimicing the standard reach environments, such as Isaac-Reach-UR10-v0 or Isaac-Reach-Franka-v0, was created for this robot specifically, however because the arm does not have as much freedom, the gripper only needs to reach a specific position, not a specific position and orientation.

The main structure of this repository is as follows:  
* so101_new_calib.urdf - Contains the URDF description of the SO-ARM101 robot from the open-source repository above. This is used to determine joint-limits for the robot, as LeRobot functions with normalized values [-100, 100] for joint positions ([0, 100] in the case of the gripper).  
  
* so101_joint_states.py - Connects to the robot and prints out the current joint states, and end-effector pose. Useful for debugging and verifying that the agent performed correctly.

* so101_reach.py - The implementation of the trained agent mentioned above, for this specific task. This script initializes the robot interfaces, agent, and performs all the integration between the two.
