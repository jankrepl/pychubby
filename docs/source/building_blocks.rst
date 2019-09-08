.. _building_blocks:

Building Blocks
===============
This page is dedicated to explaining the logic behind :code:`pychubby`.

68 landmarks
------------

LandmarkFace
------------

.. _building_blocks_reference_space:

ReferenceSpace
--------------
In general, faces in images appear in different positions, angles and sizes. Defining actions purely
based on coordinates of a given face in a given image is not a great idea. Mainly for two reasons:

	1) Resizing, cropping, rotating, etc of the image will render the action useless
	2) These actions are image specific and cannot be applied to any other image. One would be
	   better off using some graphical interface.

One way to solve the above issues is to first transform all the landmarks into some reference space,
define actions in this reference space and then map it back into the original domain. :code:`pychubby`
defines these reference spaces in :code:`pychubby.reference` module. Each reference space needs to implement
the following three methods:

- :code:`estimate`
- :code:`inp2ref`
- :code:`ref2inp`

The default reference space is the :code:`DefaultRS` and its logic is captured in the below figure.

.. image:: https://i.imgur.com/HRBKTr4.gif 
   :width: 800
   :align: center


Five selected landmarks are used to estimate an affine transformation between the reference and input space.
This trasformation is endcoded in a 2 x 3 matrix **A**. Transforming from reference to input space
and vice versa is then just a simple matrix multiplication.

.. math::

	\textbf{x}_{inp}A = \textbf{x}_{ref}

.. math::

	\textbf{x}_{ref}A^{-1} = \textbf{x}_{inp}


Action
------

DisplacementField
-----------------
