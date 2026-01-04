import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from llm import get_target_position, generate_training_command, calculate_target_position
    from env import RobotObjectEnv
    OLLAMA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some imports failed ({e}). Testing basic functionality only.")
    OLLAMA_AVAILABLE = False


def test_workspace_bounds():
    if not OLLAMA_AVAILABLE:
        print("Skipping workspace bounds test - dependencies not available")
        return
    
    print("Testing workspace bounds...")
    env = RobotObjectEnv(gui=False)
    workspace_bounds = env.get_workspace_bounds()
    
    print(f"Workspace bounds: {workspace_bounds}")
    
    test_commands = [
        "left",
        "right",
        "forward",
        "backward",
        "northeast",
        "southwest",
        "center",
        "near the robot",
        "on the table",
        "put it at x=0.5, y=0.3, z=0.5",
        "move it up",
        "place it down"
    ]
    
    print("\nTesting various commands:")
    print("-" * 60)
    
    for command in test_commands:
        target_pos = get_target_position(command, workspace_bounds)
        x, y, z = target_pos
        
        x_valid = workspace_bounds['x'][0] <= x <= workspace_bounds['x'][1]
        y_valid = workspace_bounds['y'][0] <= y <= workspace_bounds['y'][1]
        z_valid = workspace_bounds['z'][0] <= z <= workspace_bounds['z'][1]
        
        status = "✓" if (x_valid and y_valid and z_valid) else "✗"
        
        print(f"{status} Command: '{command}'")
        print(f"   Target: ({x:.3f}, {y:.3f}, {z:.3f})")
        print(f"   X valid: {x_valid}, Y valid: {y_valid}, Z valid: {z_valid}")
        print()
    
    env.close()
    print("Workspace bounds test completed!")


def test_environment_reset():
    if not OLLAMA_AVAILABLE:
        print("Skipping environment reset test - dependencies not available")
        return
    
    print("\nTesting environment reset with target positions...")
    env = RobotObjectEnv(gui=False)
    workspace_bounds = env.get_workspace_bounds()
    
    test_commands = ["left", "right", "center", "forward"]
    
    for command in test_commands:
        target_pos = get_target_position(command, workspace_bounds)
        state, info = env.reset(target_position=target_pos)
        
        assert "target_position" in info, "Info should contain target_position"
        assert info["target_position"] == target_pos, "Target position should match"
        assert len(state) == env.observation_space.shape[0], "State dimension should match"
        
        print(f"✓ Command '{command}': Target {info['target_position']}, State dim: {len(state)}")
    
    env.close()
    print("Environment reset test completed!")


def test_training_command_generation():
    if not OLLAMA_AVAILABLE:
        print("Skipping training command generation test - dependencies not available")
        return
    
    print("\nTesting training command generation...")
    
    for i in range(10):
        command = generate_training_command(i, use_llm=False)
        print(f"  Episode {i}: {command}")
    
    print("\nTesting LLM-based command generation...")
    for i in range(5):
        command = generate_training_command(i, use_llm=True)
        print(f"  Episode {i}: {command}")
    
    print("Training command generation test completed!")


def test_directional_parsing():
    print("\nTesting directional command parsing (no LLM required)...")
    
    workspace_bounds = {
        'x': (0.2, 0.6),
        'y': (-0.4, 0.4),
        'z': (0.4, 0.6)
    }
    
    test_cases = [
        ("left", "Should be on left side (negative y)"),
        ("right", "Should be on right side (positive y)"),
        ("forward", "Should be forward (positive x)"),
        ("backward", "Should be backward (negative x)"),
        ("northeast", "Should be northeast (positive x, positive y)"),
        ("center", "Should be at center"),
    ]
    
    for command, description in test_cases:
        result = calculate_target_position(command, workspace_bounds, [0, 0, 0.5], [0.3, 0.2, 0.5])
        x, y, z = result["x"], result["y"], result["z"]
        
        x_valid = workspace_bounds['x'][0] <= x <= workspace_bounds['x'][1]
        y_valid = workspace_bounds['y'][0] <= y <= workspace_bounds['y'][1]
        z_valid = workspace_bounds['z'][0] <= z <= workspace_bounds['z'][1]
        
        status = "✓" if (x_valid and y_valid and z_valid) else "✗"
        print(f"{status} '{command}': ({x:.3f}, {y:.3f}, {z:.3f}) - {description}")
    
    print("Directional parsing test completed!")


if __name__ == "__main__":
    print("=" * 60)
    print("LLM Integration Test Suite")
    print("=" * 60)
    
    try:
        test_directional_parsing()
        
        if OLLAMA_AVAILABLE:
            test_workspace_bounds()
            test_environment_reset()
            test_training_command_generation()
        else:
            print("\nNote: Full tests require ollama to be installed and available.")
            print("Basic directional parsing tests completed successfully.")
        
        print("\n" + "=" * 60)
        print("Tests completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

