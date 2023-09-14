Validate Installation
=====================

Bullet Environment validation
```shell
python3 test/bullet_env_init.py
```

If you have successfully run ROS and Gazebo, you can run ROS environment

```shell
python3 test/ros_env_init.py
```

Running ROS and Gazebo
======================

Run following command to run ROS with simulation launch file
```shell
test-franka-simulation-full camera:=true
```

After simulation initialised, run RVIZ
```shell
test-franka-desktop-rviz
```
