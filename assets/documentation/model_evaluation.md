# Test and Evaluate Trained Model

You can test ROS or Bullet environments with trained models<br>
```shell
python3 test/ros_experiment_trained_model_position.py
# OR If you have ROS running
python3 test/ros_experiment_trained_model_orientation.py
```

To test the models on Bullet simulation, you can basically replace `RoboRL-Navigator-Panda-ROS` with `RoboRL-Navigator-Panda-ROS`.

# Comparison with Classical Methods

```shell
python3 test/classical_metohds_comparison/evaluation_all_methods.py
```
To view graph
```shell
assets/evaluation_results/evaluation_results_graph_with_range.py
```

