import pandas as pd
import matplotlib.pyplot as plt


# Read the CSV file into a DataFrame
data = pd.read_csv('/models/roborl-navigator/model_comparison/TD3_HER_50K/logs/progress.csv')

# Extract the 'rollout/success_rate' column
success_rate = data['rollout/success_rate']

plt.figure(figsize=(10, 6))
plt.plot(success_rate, marker='o', linestyle='-')

plt.xlabel('Time/Episodes')
plt.ylabel('Rollout Success Rate')
plt.title('Rollout Success Rate Over Time/Episodes')

# Show the plot
plt.grid(True)
plt.show()
