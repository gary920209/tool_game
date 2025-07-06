prompt_init = """
You are an expert in tool-based physical reasoning tasks using the pyGameWorld environment.

**MAIN TASK:**
Your goal is to help a red ball reach a green container by placing one rigid body tool in the scene. The ball starts in a static (stationary) position, and you need to strategically place a rigid tool above the ball to collide with the ball and direct its path to the goal container.

You will be given an initial scene of a 2D physics simulation. The environment is affected by gravity, and your goal is to solve the task by placing one rigid body tool into the scene.

In the top-right corner of the image, you will see three predefined rigid body tools displayed for reference. These are your only choices. Each tool has a label under it:
- obj1: a trapezoid-shaped rigid body
- obj2: a diamond-shaped rigid body  
- obj3: a long flat rectangular rigid plank

Your task is to choose **one** of these rigid body tools, and place it somewhere in the main scene (not the top-right corner). You should carefully analyze the scene and use the tool's rigid body properties and physics to collide with the red ball and guide it into the green container.

The canvas is 600x600 pixels. The placement action is a pair of pixel coordinates: [x, y], where (0,0) is bottom-left and (600,600) is top-right. Tools should be placed in areas not overlapping with other objects or outside the scene.

Your action should be in the following format:
{
  "analysis": "<analysis>",          # describe the reason why you choose this tool and where to place it
  "toolname": "<tool_name>",         # choose from "obj1", "obj2", or "obj3"
  "position": [x, y]                 # where to place the center of the tool, in pixel coordinates
}

First, describe your analysis of the scene: what is happening, what goal needs to be achieved, and what physical principles may help (e.g., gravity, momentum, rigid body collisions, impact angles).

Then, propose your rigid body tool placement in the JSON format as described above.
"""


prompt_check = """
**MAIN TASK:**
Your goal is to help a red ball reach a green container by placing one rigid body tool in the scene. The ball starts in a static (stationary) position, and you need to strategically place a rigid tool over the ball to collide with the ball and direct its path to the goal container.
You previously placed the rigid body tool <PREDICTED_ACTION> in the scene.

Now, I will show you the scene with that rigid body tool placement.

Please:
1. Describe what you observe in the new image.
2. Reflect on your previous strategy: was your rigid body tool positioned correctly to collide with the ball? Why or why not?
3. If you think the rigid body tool placement was effective for collision and redirection, return the same action.
4. If not, propose a new action in the format:
{
  "analysis": "<analysis>",          # describe the reason why you choose this tool and where to place it
  "toolname": "<tool_name>",         
  "position": [x, y]                 
}
"""

prompt_feedback = """
**MAIN TASK:**
Your goal is to help a red ball reach a green container by placing one rigid body tool in the scene. The ball starts in a static (stationary) position, and you need to strategically place a rigid tool over the ball to collide with the ball and direct its path to the goal container.

You previously placed the rigid body tool <PREDICTED_ACTION> in the scene, but the task was not solved.

The images you are seeing show the COMPLETE SIMULATION PROCESS from start to finish after your rigid body tool placement. These images capture key moments throughout the physics simulation, showing how the ball moved, interacted with your tool, and where it ended up.

Please analyze this complete simulation sequence:
1. What happened to the red ball throughout the simulation? Did it collide with your rigid body tool as intended?
2. If the collision did not happen, try to modified the rigid body tool placement (maybe replace it more to the left or right) to make it collide with the ball.
3. If the collision happened, but the ball did not reach the goal container, try to modified the rigid body tool placement to make it reach the goal container.

Based on this analysis of the entire simulation process, propose a new action that addresses the issues you observed:
{
  "analysis": "<analysis>",          # describe the reason why you choose this tool and where to place it
  "toolname": "<tool_name>",         
  "position": [x, y]                 
}

Think about:
- Would a different rigid body tool work better for this specific collision physics problem?
- Should the rigid body tool be placed in a different location to better collide with the ball?
- Are there other physics principles you can leverage (momentum transfer, collision angles, rigid body dynamics)?
- How can you better control the ball's trajectory through rigid body collisions?
- What do the simulation images reveal about the timing and effectiveness of the collision?
"""

prompt_invalid = """
The predicted rigid body tool placement <PREDICTED_ACTION> is invalid — the rigid body tool may be outside the scene or overlapping with other objects on the scene.

Please re-analyze the scene and propose a corrected rigid body tool placement, modified the position of the tool to the left or right or up or down.

Your response should follow this format:
{
  "analysis": "<analysis>",          # describe the reason why you choose this tool and where to place it
  "toolname": "<tool_name>",         
  "position": [x, y]                 
}
"""


