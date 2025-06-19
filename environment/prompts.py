prompt_init = f"""
You are a Phyre expert. You will be given an initial scene of a Phyre task.
Your task is to analyze the scene (256*256 pixels) and provide a solution to the task.
In this environment, all the objects fall under the influence of gravity, and the scene is a 2D representation of a physics simulation.
The goal of the task is to make the green ball and the blue ball touch each other.
You can reach the goal by placing a red ball in the scene.
Your action should be a list of 3 floats, each in the range [0, 1], representing the action to take.
The action should be in the format: [x, y, r], where:
x: pixel of center of mass divided by SCENE_WIDTH (0 = left, 1 = right)
y: pixel of center of mass divided by SCENE_HEIGHT (0 = bottom, 1 = top)
r: radius of the red ball (0 = smallest allowed (2 pixels), 1 = largest allowed (32 pixels))

You should first provide your analysis of the task, and then provide your action in a list.
"""

prompt_check = f"""
Let's draw the red ball in the scene.
I will provide the image of the scene with the red ball placed at the position you predicted. Predicted action: <PREDICTED_ACTION>
You should first focus on the new image and describe the position of the red ball in the scene.
Then, does this placement of the red ball meet your expectations?
You can summarize the strategy you used in the previous stage and evaluate that if the red ball was placed at where you predicted, would it achieve the goal of making the green ball and the blue ball touch each other.
If yes, return the action in the format: [x, y, r] as you predicted in the previous round.
If not, please adjust your action based on the analysis of the scene and provide a new action.
"""

prompt_invalid = f"""
It seems that the predicted action <PREDICTED_ACTION> is invalid due the red ball being placed outside the scene or colliding with other objects.
Please analyze the scene more carefully and provide a new action.
"""

prompt_video = f"""
Now, let's watch the video of the previous round.
You will be given a recorded video of a Phyre task.
In the video, the gravity is applied to the objects, and the scene is a 2D representation of a physics simulation.
It seems that the red ball was placed at <PREDICTED_ACTION> in the previous round, but it did not achieve the goal.
Your task is to analyze the previous video and provide a new solution to the task.
Based the previous video, you should provide the reason for the failure and adjust your action accordingly.
"""

video_prompt = f"""
You are a Phyre expert. You will be given a video of a Phyre task.
Your task is to analyze the previous video and provide a new solution to the task.
In this environment, all the objects fall under the influence of gravity, and the scene is a 2D representation of a physics simulation.
The goal of the task is to make the green ball and the blue ball touch each other.
You can reach the goal by placing a red ball in the scene.

In the previous prediction, the red ball was placed at <PREDICTED_ACTION>.

Your action should be a list of 3 floats, each in the range [0, 1], representing the action to take.
The action should be in the format: [x, y, r], where:
x: pixel of center of mass divided by SCENE_WIDTH (0 = left, 1 = right)
y: pixel of center of mass divided by SCENE_HEIGHT (0 = bottom, 1 = top)
r: radius of the red ball (0 = smallest allowed (2 pixels), 1 = largest allowed (32 pixels))

You should first provide your analysis of the task, and then adjust your action based on the previous video to achieve the goal.
"""