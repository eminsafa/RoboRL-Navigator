Grasping Pose Detection (GPD)
=============================

.. code:: shell

   conda activate contact_graspnet_env
   python contact_graspnet/contact_graspnet_server.py

View Grasping Poses
-------------------

API Reference for GPD Server
----------------------------

Local Configuration
~~~~~~~~~~~~~~~~~~~

.. code:: http

     GET /run

Request
^^^^^^^

========= ========== ========================
Parameter Type       Description
========= ========== ========================
``path``  ``string`` **Required**. Image data
========= ========== ========================

Response
^^^^^^^^

========= ======== =============================
Parameter Type     Description
========= ======== =============================
``file``  ``FILE`` **Required**. predictions.npz
========= ======== =============================

LAN Configuration
~~~~~~~~~~~~~~~~~

.. code:: http

     GET /run

.. _request-1:

Request
^^^^^^^

========= ======== ===========================
Parameter Type     Description
========= ======== ===========================
``file``  ``FILE`` **Required**. data.npy file
========= ======== ===========================

.. _response-1:

Response
^^^^^^^^

========= ======== =============================
Parameter Type     Description
========= ======== =============================
``file``  ``FILE`` **Required**. predictions.npz
========= ======== =============================
