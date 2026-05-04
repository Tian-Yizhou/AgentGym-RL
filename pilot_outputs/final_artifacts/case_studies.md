# Pilot Case Studies

## Summary

- BabyAI benchmark reached non-trivial rewards and produced many solved or partially solved trajectories.
- TextWorldExpress ran end-to-end but exposes action-grounding failures: the model often invents plausible actions that are not in the valid-action list.
- A small TextWorldExpress client postprocessor reduced `Unknown action` count from 54 to 47 on the smoke setting.

## BabyAI High-Reward Example

- Source: `saves/babyai_benchmark_7099097/executer_logs/step65/0.json`
- Reward: `0.9984375`

- **user**: You are an exploration master that wants to finish every goal you are given. Every round I will give you an observation, and you have to respond an action and your thought based on the observation to finish the given task. You are placed in a room and you need to accomplish the given goal with actions. You can use the following actions: - turn r...
- **assistant**: OK. I'll follow your instructions and try my best to solve the task.
- **user**: Your goal: go to the purple ball In front of you in this room, you can see several objects: There is a purple ball 1 right in front of you 2 steps away. There is a blue box 1 0 steps in front of you and 2 steps to your right. There is a purple key 1 right in front of you 3 steps away. There is a green ball 1 2 steps in front of you and 1 steps t...
- **assistant**: Thought: I need to go to the purple ball. Action: go to purple ball 1
- **user**: In front of you in this room, you can see several objects: There is a purple ball 1 right in front of you 1 steps away. There is a purple key 1 right in front of you 2 steps away. There is a green ball 1 1 steps in front of you and 1 steps to your left. There is a blue ball 1 1 steps in front of you and 1 steps to your right. There is a green bo...

## BabyAI Low-Reward / Failure Example

- Source: `saves/babyai_benchmark_7099097/executer_logs/step1/0.json`
- Reward: `0`

- **user**: You are an exploration master that wants to finish every goal you are given. Every round I will give you an observation, and you have to respond an action and your thought based on the observation to finish the given task. You are placed in a room and you need to accomplish the given goal with actions. You can use the following actions: - turn r...
- **assistant**: OK. I'll follow your instructions and try my best to solve the task.
- **user**: Your goal: go to the red ball In front of you in this room, you can see several objects: The room has walls around you. You are facing a wall 3 steps away. You are not carrying anything. Available actions: ["turn left", "turn right", "move forward", "check available actions"]
- **assistant**: Thought: I need to move forward and check if any of the available actions can turn me to another room or object. Action: move forward
- **user**: In front of you in this room, you can see several objects: The room has walls around you. You are facing a wall 2 steps away. You are not carrying anything. Available actions: ["turn left", "turn right", "move forward", "check available actions"]
- **assistant**: Thought: I need to check my position to see if there is any door or other objects in front of me. Action: check available actions
- **user**: You can take the following actions: turn left, turn right, move forward, check available actions Available actions: ["turn left", "turn right", "move forward", "check available actions"]
- **assistant**: Thought: I need to check for any door or other objects in front of me. Action: check available actions
- **user**: You can take the following actions: turn left, turn right, move forward, check available actions Available actions: ["turn left", "turn right", "move forward", "check available actions"]
- **assistant**: Thought: I also need to check for the red ball and if it can be seen in front of me. Action: check available actions
- ... 9 more messages

## TextWorldExpress Action-Grounding Example

- Source: `saves/textworldexpress_benchmark_7111047/executer_logs/step1/0.json`
- Reward: `0.0`

- **user**: You are playing a text adventure game. Every round I will give you an observation and a list of valid actions. Think briefly, then choose exactly one action copied verbatim from the valid actions list. Your output must strictly follow this format: Thought: <your thoughts> Action: <one valid action>
- **assistant**: OK. I will choose one valid action each round.
- **user**: You are in the bedroom. In one part of the room you see a dressing table, that has nothing on it. There is also a desk chair, that has nothing on it. You also see a desk, that has nothing on it. In another part of the room you see a chest of drawers that is closed. In one part of the room you see a wardrobe that is closed. There is also a night ...
- **assistant**: Thought: In the bedroom, the only option to the west is the closed wood door. Action: open door to west
- **user**: You open the wood door, revealing the living room. Valid actions: - close door to west - inventory - look around - move west - open chest of drawers - open door to west - open wardrobe - take used tissue
- **assistant**: Thought: There are no openings to the open door on the west. Action: close door to west
- **user**: You close the wood door to the living room. Valid actions: - close door to west - inventory - look around - move west - open chest of drawers - open door to west - open wardrobe - take used tissue
- **assistant**: Thought: Internal rooms like the living room do not have windows or doors, so no opening to the west is available. Action: close door to west
- **user**: That is already closed. Valid actions: - close door to west - inventory - look around - move west - open chest of drawers - open door to west - open wardrobe - take used tissue
- **assistant**: Thought: The closest option that could open the door to the west is to try to open the chest of drawers. Action: open chest of drawers
- ... 5 more messages
