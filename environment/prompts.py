prompt_init = """
You are an expert in tool-based physical reasoning tasks using the pyGameWorld environment.

You will be given an initial scene of a 2D physics simulation. The environment is affected by gravity, and your goal is to solve the task by placing one tool into the scene.

In the top-right corner of the image, you will see three predefined red tools displayed for reference. These are your only choices. Each tool has a label under it:
- obj1: a long L-shaped tool
- obj2: a diamond-shaped tool
- obj3: a long flat rectangular plank

Your task is to choose **one** of these tools, and place it somewhere in the main scene (not the top-right corner). You should carefully analyze the scene and use the tool's shape and physics to help the blue ball reach the green container.

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

prompt_invalid = """
The predicted tool placement <PREDICTED_ACTION> is invalid — the tool may be outside the scene or overlapping with other objects.

Please re-analyze the scene and propose a corrected tool placement.

Your response should follow this format:
{
  "toolname": "<tool_name>",         
  "position": [x, y]                 
}
"""

prompt_video = """
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

video_prompt = """
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
