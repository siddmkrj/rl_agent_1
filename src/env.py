import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RobotObjectEnv:
    def __init__(self, gui=True):
        self.gui = gui
        if gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        self.plane_id = p.loadURDF("plane.urdf")
        
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0.5])
        
        self.object_id = p.loadURDF("cube_small.urdf", [0.3, 0.2, 0.5])
        
        num_joints = p.getNumJoints(self.robot_id)
        self.num_joints = num_joints
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_joints,), dtype=np.float32
        )
        
        obs_dim = num_joints * 2 + 3 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.target_position = None
        self.max_steps = 200
        self.current_step = 0
        
        self.workspace_bounds = {
            'x': (0.2, 0.6),
            'y': (-0.4, 0.4),
            'z': (0.4, 0.6)
        }
        
    def reset(self, seed=None, options=None, target_position=None):
        if seed is not None:
            np.random.seed(seed)
        
        p.resetBasePositionAndOrientation(
            self.robot_id, [0, 0, 0.5], [0, 0, 0, 1]
        )
        
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, 0, 0)
        
        object_x = np.random.uniform(0.2, 0.4)
        object_y = np.random.uniform(-0.2, 0.2)
        p.resetBasePositionAndOrientation(
            self.object_id, [object_x, object_y, 0.5], [0, 0, 0, 1]
        )
        
        if target_position is None:
            x_min, x_max = self.workspace_bounds['x']
            y_min, y_max = self.workspace_bounds['y']
            z_min, z_max = self.workspace_bounds['z']
            target_position = [
                np.random.uniform(x_min, x_max),
                np.random.uniform(y_min, y_max),
                np.random.uniform(z_min, z_max)
            ]
        
        self.target_position = self._validate_target_position(target_position)
        self.current_step = 0
        
        observation = self._get_observation()
        info = {"target_position": self.target_position}
        
        return observation, info
    
    def step(self, action):
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.VELOCITY_CONTROL,
                targetVelocity=action[i] * 2.0,
                force=50.0
            )
        
        p.stepSimulation()
        
        self.current_step += 1
        
        observation = self._get_observation()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        
        info = {"target_position": self.target_position}
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        joint_states = []
        for i in range(self.num_joints):
            joint_info = p.getJointState(self.robot_id, i)
            joint_states.append(joint_info[0])
            joint_states.append(joint_info[1])
        
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)
        robot_euler = p.getEulerFromQuaternion(robot_orn)
        
        object_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        
        target_pos = self.target_position
        
        observation = np.array(
            joint_states + list(robot_pos) + list(object_pos) + list(target_pos),
            dtype=np.float32
        )
        
        return observation
    
    def _compute_reward(self):
        object_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        
        target_pos = self.target_position
        
        distance_to_object = np.linalg.norm(
            np.array(robot_pos) - np.array(object_pos)
        )
        
        distance_to_target = np.linalg.norm(
            np.array(object_pos) - np.array(target_pos)
        )
        
        reward = -distance_to_object * 0.1 - distance_to_target * 0.5
        
        if distance_to_target < 0.1:
            reward += 10.0
        
        if distance_to_object < 0.15:
            reward += 2.0
        
        return reward
    
    def _is_terminated(self):
        object_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        
        target_pos = self.target_position
        
        distance_to_target = np.linalg.norm(
            np.array(object_pos) - np.array(target_pos)
        )
        
        return distance_to_target < 0.1
    
    def _validate_target_position(self, target_position):
        x, y, z = target_position
        x_min, x_max = self.workspace_bounds['x']
        y_min, y_max = self.workspace_bounds['y']
        z_min, z_max = self.workspace_bounds['z']
        
        x = max(x_min, min(x_max, x))
        y = max(y_min, min(y_max, y))
        z = max(z_min, min(z_max, z))
        
        return [x, y, z]
    
    def get_workspace_bounds(self):
        return self.workspace_bounds.copy()
    
    def close(self):
        p.disconnect(self.physics_client)

