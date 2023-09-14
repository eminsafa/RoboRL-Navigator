Train Your Model
================

You can train your model as shown in example
```shell
python3 train/examples/bullet_training.py
# OR If you have ROS running
python3 train/examples/ros_training.py
```
You can check environment parameters [here](environments.md).<br>

Your trained model will be saved in `~/RoboRL-Navigator/models/roborl-navigator/` directory.
Please note that, git does not track this directory as file size is over the limit.

Download Trained Model
======================

You can install pre-trained model [here](https://drive.google.com/file/d/1EMeIu4W3FPgGrlhQ_Q8RUQBGgJ0cb7uQ/view?usp=sharing).
Extract downloaded model to `models/` directory, It will look like this:

.. code:: shell

    external
    └── models
       └── roborl-navigator
           └── TD3_Bullet_0.05_Threshold_200K
               ├── model.zip

