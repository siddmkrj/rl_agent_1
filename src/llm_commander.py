from llm_integration import LLMIntegration
from typing import Dict, List, Optional, Tuple
import numpy as np
import random


def get_target_position(command: str, workspace_bounds: Dict[str, Tuple[float, float]],
                       robot_position: Optional[List[float]] = None,
                       object_position: Optional[List[float]] = None) -> List[float]:
    """
    Convert natural language command to 3D target position.
    
    Args:
        command: Natural language command (e.g., "put it on the left", "move forward", "at x=0.5, y=0.3")
        workspace_bounds: Dict with 'x', 'y', 'z' keys, each containing (min, max) tuple
        robot_position: Current robot position [x, y, z] (optional)
        object_position: Current object position [x, y, z] (optional)
    
    Returns:
        Target position as [x, y, z] within workspace bounds
    """
    llm = LLMIntegration()
    
    if robot_position is None:
        robot_position = [0.0, 0.0, 0.5]
    if object_position is None:
        object_position = [0.3, 0.2, 0.5]
    
    command_lower = command.lower().strip()
    
    result = llm.interpret_command(command, workspace_bounds, robot_position, object_position)
    
    if result["success"]:
        coordinates = llm.extract_coordinates(result["raw_response"], workspace_bounds)
        if coordinates:
            return coordinates
    
    return _parse_directional_command(command_lower, workspace_bounds, robot_position, object_position)


def _parse_directional_command(command: str, workspace_bounds: Dict[str, Tuple[float, float]],
                              robot_position: List[float], object_position: List[float]) -> List[float]:
    """
    Parse directional and natural language commands to coordinates.
    Falls back to default positions if command is ambiguous.
    """
    x_min, x_max = workspace_bounds['x']
    y_min, y_max = workspace_bounds['y']
    z_min, z_max = workspace_bounds['z']
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    if "left" in command:
        target_y = y_min + y_range * 0.25
        return [x_center, target_y, z_center]
    
    if "right" in command:
        target_y = y_max - y_range * 0.25
        return [x_center, target_y, z_center]
    
    if "forward" in command or "front" in command:
        target_x = x_max - x_range * 0.25
        return [target_x, y_center, z_center]
    
    if "backward" in command or "back" in command or "behind" in command:
        target_x = x_min + x_range * 0.25
        return [target_x, y_center, z_center]
    
    if "north" in command:
        if "east" in command:
            return [x_max - x_range * 0.25, y_max - y_range * 0.25, z_center]
        elif "west" in command:
            return [x_min + x_range * 0.25, y_max - y_range * 0.25, z_center]
        else:
            return [x_center, y_max - y_range * 0.25, z_center]
    
    if "south" in command:
        if "east" in command:
            return [x_max - x_range * 0.25, y_min + y_range * 0.25, z_center]
        elif "west" in command:
            return [x_min + x_range * 0.25, y_min + y_range * 0.25, z_center]
        else:
            return [x_center, y_min + y_range * 0.25, z_center]
    
    if "east" in command:
        return [x_max - x_range * 0.25, y_center, z_center]
    
    if "west" in command:
        return [x_min + x_range * 0.25, y_center, z_center]
    
    if "up" in command or "above" in command:
        target_z = min(z_max, z_center + (z_max - z_center) * 0.5)
        return [x_center, y_center, target_z]
    
    if "down" in command or "below" in command:
        target_z = max(z_min, z_center - (z_center - z_min) * 0.5)
        return [x_center, y_center, target_z]
    
    if "near" in command or "close" in command:
        if robot_position:
            offset = 0.1
            target_x = robot_position[0] + random.uniform(-offset, offset)
            target_y = robot_position[1] + random.uniform(-offset, offset)
            target_z = robot_position[2]
            target_x = max(x_min, min(x_max, target_x))
            target_y = max(y_min, min(y_max, target_y))
            target_z = max(z_min, min(z_max, target_z))
            return [target_x, target_y, target_z]
    
    if "on the table" in command or "table" in command:
        return [x_center, y_center, z_min + 0.05]
    
    if "center" in command or "middle" in command:
        return [x_center, y_center, z_center]
    
    return [x_center, y_center, z_center]


def generate_training_command(episode: int, use_llm: bool = False) -> str:
    """
    Generate a command for training. Can use LLM or predefined commands.
    """
    if use_llm:
        commands = [
            "put the object on the left side",
            "move it to the right",
            "place it forward",
            "put it near the robot",
            "move it to the center",
            "place it on the table",
            "move it backward",
            "put it northeast",
            "place it southwest",
        ]
        return random.choice(commands)
    else:
        directional_commands = [
            "left", "right", "forward", "backward", 
            "north", "south", "east", "west",
            "northeast", "northwest", "southeast", "southwest",
            "center", "near the robot", "on the table"
        ]
        return random.choice(directional_commands)

