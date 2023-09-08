import json

import matplotlib.pyplot as plt
import numpy as np

file = open("all_evaluation_1.json", "r+")
results_raw = json.loads(file.read())

# Create the line graph
plt.figure(figsize=(10, 6))

results = []
for ep, res in results_raw.items():
    if 'rrt' in res:
        results.append(res)
episodes = list(range(1, len(results) + 1))
# Calculate minimum, maximum, and mean times for each episode
for planner in ['rrt']:
    min_times = [episode[planner]['min'] for episode in results]
    max_times = [max(episode[planner]['all'] or [1000]) for episode in results]
    mean_times = [episode[planner]['mean'] for episode in results]

    # Plot the mean computation times as a line
    plt.plot(episodes, mean_times, label=f'RRTConnect Mean', marker='o')

    # Fill the range between minimum and maximum times
    plt.fill_between(episodes, min_times, max_times, alpha=0.2, label='RRTConnect Range (min-max)')

plt.plot(episodes, [episode['rl']['total'] for episode in results], label=f'Reinforcement Learning', marker='^')

# Add labels and a legend
plt.xlabel('Episode')
plt.ylabel('Computation Time (ms)')
plt.title('Computation Time Range for Different Episodes')
plt.legend()

# Show the graph
plt.grid(True)
plt.tight_layout()
plt.show()
