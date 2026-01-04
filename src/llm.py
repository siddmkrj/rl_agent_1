from typing import Dict, List, Optional, Tuple
import json
import random

try:
    from ollama import chat
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    chat = None


def calculate_target_position(
    command: str,
    workspace_bounds: Dict[str, Tuple[float, float]],
    robot_position: Optional[List[float]] = None,
    object_position: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Calculate target position for object placement based on natural language command.
    
    Args:
        command: Natural language command (e.g., "put it on the left", "move forward")
        workspace_bounds: Dict with 'x', 'y', 'z' keys, each containing (min, max) tuple
        robot_position: Current robot position [x, y, z] (optional)
        object_position: Current object position [x, y, z] (optional)
    
    Returns:
        Dictionary with 'x', 'y', 'z' keys containing target position coordinates
    """
    if robot_position is None:
        robot_position = [0.0, 0.0, 0.5]
    if object_position is None:
        object_position = [0.3, 0.2, 0.5]
    
    x_min, x_max = workspace_bounds['x']
    y_min, y_max = workspace_bounds['y']
    z_min, z_max = workspace_bounds['z']
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    command_lower = command.lower().strip()
    
    if "left" in command_lower:
        return {"x": x_center, "y": y_min + y_range * 0.25, "z": z_center}
    if "right" in command_lower:
        return {"x": x_center, "y": y_max - y_range * 0.25, "z": z_center}
    if "forward" in command_lower or "front" in command_lower:
        return {"x": x_max - x_range * 0.25, "y": y_center, "z": z_center}
    if "backward" in command_lower or "back" in command_lower or "behind" in command_lower:
        return {"x": x_min + x_range * 0.25, "y": y_center, "z": z_center}
    if "northeast" in command_lower:
        return {"x": x_max - x_range * 0.25, "y": y_max - y_range * 0.25, "z": z_center}
    if "northwest" in command_lower:
        return {"x": x_min + x_range * 0.25, "y": y_max - y_range * 0.25, "z": z_center}
    if "southeast" in command_lower:
        return {"x": x_max - x_range * 0.25, "y": y_min + y_range * 0.25, "z": z_center}
    if "southwest" in command_lower:
        return {"x": x_min + x_range * 0.25, "y": y_min + y_range * 0.25, "z": z_center}
    if "north" in command_lower:
        return {"x": x_center, "y": y_max - y_range * 0.25, "z": z_center}
    if "south" in command_lower:
        return {"x": x_center, "y": y_min + y_range * 0.25, "z": z_center}
    if "east" in command_lower:
        return {"x": x_max - x_range * 0.25, "y": y_center, "z": z_center}
    if "west" in command_lower:
        return {"x": x_min + x_range * 0.25, "y": y_center, "z": z_center}
    if "up" in command_lower or "above" in command_lower:
        return {"x": x_center, "y": y_center, "z": min(z_max, z_center + (z_max - z_center) * 0.5)}
    if "down" in command_lower or "below" in command_lower:
        return {"x": x_center, "y": y_center, "z": max(z_min, z_center - (z_center - z_min) * 0.5)}
    if "near" in command_lower or "close" in command_lower:
        offset = 0.1
        target_x = max(x_min, min(x_max, robot_position[0] + random.uniform(-offset, offset)))
        target_y = max(y_min, min(y_max, robot_position[1] + random.uniform(-offset, offset)))
        target_z = max(z_min, min(z_max, robot_position[2]))
        return {"x": target_x, "y": target_y, "z": target_z}
    if "table" in command_lower:
        return {"x": x_center, "y": y_center, "z": z_min + 0.05}
    if "center" in command_lower or "middle" in command_lower:
        return {"x": x_center, "y": y_center, "z": z_center}
    
    return {"x": x_center, "y": y_center, "z": z_center}


def get_target_position(command: str, workspace_bounds: Dict[str, Tuple[float, float]],
                       robot_position: Optional[List[float]] = None,
                       object_position: Optional[List[float]] = None) -> List[float]:
    """
    Convert natural language command to 3D target position using FunctionGemma.
    Falls back to simple parsing if FunctionGemma is unavailable.
    """
    if not OLLAMA_AVAILABLE:
        result = calculate_target_position(command, workspace_bounds, robot_position, object_position)
        return [result["x"], result["y"], result["z"]]
    
    if robot_position is None:
        robot_position = [0.0, 0.0, 0.5]
    if object_position is None:
        object_position = [0.3, 0.2, 0.5]
    
    context = f"""Workspace: X[{workspace_bounds['x'][0]}-{workspace_bounds['x'][1]}], Y[{workspace_bounds['y'][0]}-{workspace_bounds['y'][1]}], Z[{workspace_bounds['z'][0]}-{workspace_bounds['z'][1]}]
Robot at: ({robot_position[0]:.2f}, {robot_position[1]:.2f}, {robot_position[2]:.2f})
Object at: ({object_position[0]:.2f}, {object_position[1]:.2f}, {object_position[2]:.2f})
Command: {command}

Calculate where to place the object."""
    
    messages = [{"role": "user", "content": context}]
    
    try:
        response = chat("functiongemma", messages=messages, tools=[calculate_target_position])
        
        if response.message.tool_calls:
            tool = response.message.tool_calls[0]
            if tool.function.name == "calculate_target_position":
                arguments = tool.function.arguments
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                
                result = calculate_target_position(
                    command=arguments.get("command", command),
                    workspace_bounds=workspace_bounds,
                    robot_position=robot_position,
                    object_position=object_position
                )
                
                return [result["x"], result["y"], result["z"]]
    except Exception:
        pass
    
    result = calculate_target_position(command, workspace_bounds, robot_position, object_position)
    return [result["x"], result["y"], result["z"]]


def generate_training_command(episode: int, use_llm: bool = False) -> str:
    """Generate a command for training."""
    commands = [
        "left", "right", "forward", "backward",
        "north", "south", "east", "west",
        "northeast", "northwest", "southeast", "southwest",
        "center", "near the robot", "on the table",
        "put it on the left", "move it to the right", "place it forward"
    ]
    return random.choice(commands)

