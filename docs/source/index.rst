Documentation of RoboRL Navigator
=================================

**RoboRL Navigator** is a project that provides Reinforcement Learning codebase for
manipulator robots, especially Franka Emika Panda Robot initialized.
It has Bullet and ROS Gazebo simulation environments to train the model
to reach given position. It also uses an open-source Grasping Pose Detection
project that can be tested on Gazebo Simulation or Real World.

Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.


.. note::

   This project is under active development.


.. toctree::
   :maxdepth: 2
   :caption: Install

   installation
   validate-installation

.. toctree::
   :maxdepth: 2
   :caption: Train
   environments
   train-your-model
   download-trained-model

.. toctree::
   :maxdepth: 2
   :caption: Test
   model-evaluation
   grasping-pose-detection

.. toctree::
   :maxdepth: 2
   :caption: Demonstrate
   simulation-demonstration
   real-world-demonstration
