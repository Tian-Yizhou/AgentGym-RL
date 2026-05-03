# BabyAI Pilot Case Study

## Successful heuristic case

- data_idx: `0`
- goal: `go to the red ball`
- final score: `0.8875`
- success: `True`
- turns: `1`

### Turn 0

Observation: Your goal: go to the red ball In front of you in this room, you can see several objects: There is a grey key 1 1 steps in front of you and 1 steps to your left. There is a grey ball 1 1 steps in front of you and 1 steps to your right. There is a grey key 2 2 steps in front of you and 1 steps to your left. There is a grey key 3 2 steps in front of you and 1 steps to your right. There is a grey box 1 2 steps in front of you and 2 steps to your righ...

Thought: The goal says 'go to the red ball', and the valid actions include a matching macro action: 'go to red ball 1'. I should use that direct action.

Action: `go to red ball 1`

Reward after action: `0.8875`, done: `True`

## Random-policy failure case

- data_idx: `40`
- goal: `go to the red ball`
- final score: `0.0`
- success: `False`
- turns: `8`

### Turn 0

Observation: Your goal: go to the red ball In front of you in this room, you can see several objects: There is a grey box 1 1 steps in front of you and 1 steps to your left. There is a grey box 2 0 steps in front of you and 2 steps to your left. There is a grey key 1 1 steps in front of you and 2 steps to your left. There is a grey key 2 2 steps in front of you and 2 steps to your right. There is a grey key 3 2 steps in front of you and 3 steps to your right....

Thought: I will explore using one valid action from the current action list.

Action: `move forward`

Reward after action: `0.0`, done: `False`

### Turn 1

Observation: In front of you in this room, you can see several objects: There is a grey box 1 0 steps in front of you and 1 steps to your left. There is a grey key 1 0 steps in front of you and 2 steps to your left. There is a grey key 2 1 steps in front of you and 2 steps to your right. There is a grey key 3 1 steps in front of you and 3 steps to your right. The room has walls around you. You are facing a wall 3 steps away. You are not carrying anything. Ava...

Thought: I will explore using one valid action from the current action list.

Action: `pickup grey key 3`

Reward after action: `0.0`, done: `False`

### Turn 2

Observation: In front of you in this room, you can see several objects: There is a grey key 1 1 steps in front of you and 1 steps to your left. The room has walls around you. You are facing a wall 3 steps away. You are carrying a grey key. Available actions: ["turn left", "turn right", "move forward", "drop", "go to grey key 1", "check available actions"]

Thought: I will explore using one valid action from the current action list.

Action: `turn left`

Reward after action: `0.0`, done: `False`

### Turn 3

Observation: In front of you in this room, you can see several objects: There is a grey key 1 1 steps in front of you and 1 steps to your right. There is a grey ball 1 0 steps in front of you and 3 steps to your left. There is a grey box 1 right in front of you 4 steps away. There is a grey key 2 right in front of you 5 steps away. There is a grey box 2 5 steps in front of you and 1 steps to your left. There is a grey key 3 4 steps in front of you and 3 steps...

Thought: I will explore using one valid action from the current action list.

Action: `turn right`

Reward after action: `0.0`, done: `False`

### Turn 4

Observation: In front of you in this room, you can see several objects: There is a grey key 1 1 steps in front of you and 1 steps to your left. The room has walls around you. You are facing a wall 3 steps away. You are carrying a grey key. Available actions: ["turn left", "turn right", "move forward", "drop", "go to grey key 1", "check available actions"]

Thought: I will explore using one valid action from the current action list.

Action: `go to grey key 1`

Reward after action: `0.0`, done: `False`
