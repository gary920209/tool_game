prompt_init = """
You are an expert in tool-based physical reasoning tasks using the pyGameWorld environment.

**MAIN TASK:**
Your goal is to help a red ball reach a green container by placing one tool in the scene. The ball will fall due to gravity, and you need to strategically place a tool to guide or redirect the ball's path to the goal container.

You will be given an initial scene of a 2D physics simulation. The environment is affected by gravity, and your goal is to solve the task by placing one tool into the scene.

In the top-right corner of the image, you will see three predefined red tools displayed for reference. These are your only choices. Each tool has a label under it:
- obj1: a long L-shaped tool
- obj2: a diamond-shaped tool
- obj3: a long flat rectangular plank

Your task is to choose **one** of these tools, and place it somewhere in the main scene (not the top-right corner). You should carefully analyze the scene and use the tool's shape and physics to help the red ball reach the green container.

The canvas is 600x600 pixels. The placement action is a pair of pixel coordinates: [x, y], where (0,0) is bottom-left and (600,600) is top-right. Tools should be placed in areas not overlapping with other objects or outside the scene.

Your action should be in the following format:
{
  "toolname": "<tool_name>",         # choose from "obj1", "obj2", or "obj3"
  "position": [x, y]                 # where to place the center of the tool, in pixel coordinates
}

First, describe your analysis of the scene: what is happening, what goal needs to be achieved, and what physical principles may help (e.g., gravity, momentum, lever).

Then, propose your tool placement in the JSON format as described above.
"""


prompt_check = """
**MAIN TASK:**
Your goal is to help a red ball reach a green container by placing one tool in the scene. The ball will fall due to gravity, and you need to strategically place a tool to guide or redirect the ball's path to the goal container.

You previously placed the tool <PREDICTED_ACTION> in the scene.

Now, I will show you the scene with that tool placement.

Please:
1. Describe what you observe in the new image.
2. Reflect on your previous strategy: was your tool positioned correctly? Why or why not?
3. If you think the tool placement was effective, return the same action.
4. If not, propose a new action in the format:
{
  "toolname": "<tool_name>",         
  "position": [x, y]                 
}
"""

prompt_feedback = """
**MAIN TASK:**
Your goal is to help a red ball reach a green container by placing one tool in the scene. The ball will fall due to gravity, and you need to strategically place a tool to guide or redirect the ball's path to the goal container.

You previously placed the tool <PREDICTED_ACTION> in the scene, but the task was not solved.

The image you are seeing now shows the FINAL STATE of the simulation after your tool placement. This is what actually happened when the physics simulation ran with your tool.

Please analyze this final simulation state:
1. What happened to the red ball? Did it move in the right direction?
2. Did your tool placement create the intended effect (e.g., ramp, barrier, lever)?
3. Why didn't the ball reach the green goal container?
4. What went wrong with your strategy?

Based on this analysis, propose a new action that addresses the issues you observed:
{
  "toolname": "<tool_name>",         
  "position": [x, y]                 
}

Think about:
- Would a different tool work better for this specific physics problem?
- Should the tool be placed in a different location to better guide the ball?
- Are there other physics principles you can leverage (momentum, gravity, collision angles)?
- How can you better control the ball's trajectory?
"""

prompt_invalid = """
**MAIN TASK:**
Your goal is to help a red ball reach a green container by placing one tool in the scene. The ball will fall due to gravity, and you need to strategically place a tool to guide or redirect the ball's path to the goal container.

The predicted tool placement <PREDICTED_ACTION> is invalid — the tool may be outside the scene or overlapping with other objects on the scene.

Please re-analyze the scene and propose a corrected tool placement.

Your response should follow this format:
{
  "toolname": "<tool_name>",         
  "position": [x, y]                 
}
"""

prompt_video = """
**MAIN TASK:**
Your goal is to help a red ball reach a green container by placing one tool in the scene. The ball will fall due to gravity, and you need to strategically place a tool to guide or redirect the ball's path to the goal container.

You will be shown a video of the previous simulation round.

The goal of the task is to solve a 2D physical challenge using a single tool placement. The simulation is affected by gravity and basic physics laws.

In the previous round, the tool <PREDICTED_ACTION> was placed, but the goal was not achieved.

Please analyze the simulation video and:
1. Identify why the attempt failed.
2. Adjust your strategy accordingly.
3. Propose a new action in this format:
{
  "toolname": "<tool_name>",         
  "position": [x, y]                 
}
"""

prompt_video_feedback = """
**MAIN TASK:**
Your goal is to help a red ball reach a green container by placing one tool in the scene. The ball will fall due to gravity, and you need to strategically place a tool to guide or redirect the ball's path to the goal container.

You previously placed the tool <PREDICTED_ACTION> in the scene, but the task was not solved.

The video you are seeing shows the COMPLETE SIMULATION of what happened when your tool was placed. You can see the full physics interaction, including how the ball moved, bounced, and where it ended up.

Please analyze this simulation video:
1. What was the ball's trajectory? Did it move as expected?
2. How did your tool interact with the ball? Was the interaction effective?
3. Where did the ball end up and why didn't it reach the green goal?
4. What specific physics principles were at play (momentum, gravity, collision angles)?

Based on this detailed analysis of the simulation, propose a new action that will better achieve the goal:
{
  "toolname": "<tool_name>",         
  "position": [x, y]                 
}

Consider:
- The exact trajectory the ball took and how to modify it
- The timing and positioning of tool-ball interactions
- How to better leverage physics principles
- Whether a different tool would create better interactions
"""

video_prompt = """
**MAIN TASK:**
Your goal is to help a red ball reach a green container by placing one tool in the scene. The ball will fall due to gravity, and you need to strategically place a tool to guide or redirect the ball's path to the goal container.

You are an expert in tool-based physical reasoning in the pyGameWorld environment.

You will be given a video of a physics simulation. The environment is influenced by gravity, and your task is to place one predefined tool to solve the challenge.

Previously, the tool <PREDICTED_ACTION> was used but failed.

Analyze the video and determine:
1. What went wrong?
2. How can you correct it?

Then, provide a new tool placement action in this format:
{
  "toolname": "<tool_name>",         
  "position": [x, y]                 
}
"""
