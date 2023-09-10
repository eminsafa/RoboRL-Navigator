import matplotlib.pyplot as plt
import json

file = open('/Users/safa/qa/RoboRL-Navigator/assets/evaluation_results.json', 'r+')
raw_data = json.loads(file.read())
data = []
for i in raw_data:
    if i['rrt'] < 1000 or i['prm'] < 1000:
        data.append(i)

rrt_values = [entry["rrt"] for entry in data]
prm_values = [entry["prm"] for entry in data]
rl_values = [entry["rl"] for entry in data]

# Create a line graph for "rrt" values
plt.plot(range(1, len(data) + 1), rrt_values, marker='o', label="RRT")

# Create a line graph for "prm" values
plt.plot(range(1, len(data) + 1), prm_values, marker='s', label="PRM")

# Create a line graph for "rl" values
plt.plot(range(1, len(data) + 1), rl_values, marker='^', label="RL")

# Add labels, legend, and title
plt.xlabel("Episode")
plt.ylabel("Computation Time (ms)")

# Set custom x-axis ticks for every 10 entries
x_ticks = range(1, len(data) + 1, 4)
plt.xticks(x_ticks)

plt.legend()
plt.title("Computation Times of PRM, RRT, and RL")

# Show the graph
plt.show()
