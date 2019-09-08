.. _gallery:

Gallery
=======
This page gives overview of all currently available actions.

AbsoluteMove
------------
Low-level action that allows the user to manually select the pixel distances between the old and new
landmarks. Note that this action is image specific and therefore not invariant to affine transformations.

.. code-block:: python

	AbsoluteMove(y_shifts={30: 20, 29: 20})	# move lower part of the nose 20 pixels down

.. image:: https://i.imgur.com/2ZCRCiM.gif
   :width: 300
   :align: center

Chubbify
--------
Make a face chubby. Affine transformations invariant.

.. code-block:: python

	Chubbify(0.2)	

.. image:: https://i.imgur.com/yNUNNCw.gif
   :width: 300
   :align: center

Lambda
------
Low-level action where only angles and relative sizes in the reference space are to be specified.
Affine transformation invariant. For more details see :ref:`custom_actions_lambda`.

.. code-block:: python

	Lambda(0.2, {'OUTERMOST_EYEBROW_L': (-90, 1),
		     'OUTER_EYEBROW_L': (-90, 0.8)})	

.. image:: https://i.imgur.com/xuIgqFK.gif 
   :width: 300
   :align: center

LinearTransform
---------------
Apply a linear transformation to all landmarks on a single face.

.. code-block:: python

	LinearTransform(scale_x=0.9, scale_y=0.95)

.. image:: https://i.imgur.com/s57gnkj.gif 
   :width: 300
   :align: center

Multiple
--------
Metaaction enabling handling of multiple faces in a single image.

OpenEyes
--------
Open eyes. Affine transformation invariant.

.. code-block:: python

	OpenEyes(0.06)

.. image:: https://i.imgur.com/H4kP9lI.gif 
   :width: 300
   :align: center

Pipeline
--------
Metaaction allowing for multiple actions on a single face.

.. code-block:: python

 	Pipeline([Smile(-0.08), OpenEyes(-0.06)])
	
.. image:: https://i.imgur.com/Hh6KtKa.gif 
   :width: 300
   :align: center

Smile
-----
Smile. Affine transformation invariant.

.. code-block:: python

	Smile(0.1)

.. image:: https://i.imgur.com/1oR046T.gif 
   :width: 300
   :align: center



