from typing import Dict, List, Optional, Tuple
import json

try:
    from ollama import chat
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    chat = None


class LLMIntegration:
    def __init__(self, gemma_model: str = "gemma2:2b", functiongemma_model: str = "functiongemma"):
        self.gemma_model = gemma_model
        self.functiongemma_model = functiongemma_model
    
    def interpret_command(self, command: str, workspace_bounds: Dict[str, Tuple[float, float]], 
                         robot_position: Optional[List[float]] = None,
                         object_position: Optional[List[float]] = None) -> Dict:
        if not OLLAMA_AVAILABLE:
            return {
                "raw_response": "Ollama not available",
                "success": False
            }
        
        context = f"""
You are helping a robot understand where to place an object. 
The robot workspace has the following bounds:
- X axis: {workspace_bounds['x'][0]} to {workspace_bounds['x'][1]}
- Y axis: {workspace_bounds['y'][0]} to {workspace_bounds['y'][1]}
- Z axis: {workspace_bounds['z'][0]} to {workspace_bounds['z'][1]}

"""
        if robot_position:
            context += f"Robot is currently at position: ({robot_position[0]:.2f}, {robot_position[1]:.2f}, {robot_position[2]:.2f})\n"
        if object_position:
            context += f"Object is currently at position: ({object_position[0]:.2f}, {object_position[1]:.2f}, {object_position[2]:.2f})\n"
        
        context += f"\nUser command: {command}\n"
        context += "\nInterpret this command and provide target coordinates (x, y, z) within the workspace bounds."
        
        messages = [{"role": "user", "content": context}]
        
        try:
            response = chat(self.gemma_model, messages=messages, stream=False)
            return {
                "raw_response": response.message.content,
                "success": True
            }
        except Exception as e:
            return {
                "raw_response": str(e),
                "success": False
            }
    
    def extract_coordinates(self, text: str, workspace_bounds: Dict[str, Tuple[float, float]]) -> Optional[List[float]]:
        import re
        
        patterns = [
            r'x[:\s=]+([-+]?\d*\.?\d+)[,\s]+y[:\s=]+([-+]?\d*\.?\d+)[,\s]+z[:\s=]+([-+]?\d*\.?\d+)',
            r'\(([-+]?\d*\.?\d+)[,\s]+([-+]?\d*\.?\d+)[,\s]+([-+]?\d*\.?\d+)\)',
            r'\[([-+]?\d*\.?\d+)[,\s]+([-+]?\d*\.?\d+)[,\s]+([-+]?\d*\.?\d+)\]',
            r'([-+]?\d*\.?\d+)[,\s]+([-+]?\d*\.?\d+)[,\s]+([-+]?\d*\.?\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    x = float(match.group(1))
                    y = float(match.group(2))
                    z = float(match.group(3))
                    
                    x = max(workspace_bounds['x'][0], min(workspace_bounds['x'][1], x))
                    y = max(workspace_bounds['y'][0], min(workspace_bounds['y'][1], y))
                    z = max(workspace_bounds['z'][0], min(workspace_bounds['z'][1], z))
                    
                    return [x, y, z]
                except ValueError:
                    continue
        
        return None

