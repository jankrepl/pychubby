.. _building_blocks:

Building Blocks
===============
This page is dedicated to explaining the logic behind :code:`pychubby`.

68 landmarks
------------
:code:`pychubby` relies on the standard 68 facial landmarks framework. Specifically,
a pretrained :code:`dlib` model is used to achieve this task. See :code:`pychubby.data` for credits 
and references. Once the landmarks are detected
one can query them via their index. Alternatively, for the ease of defining new actions
a dictionary :code:`pychubby.detect.LANDMARK_NAMES` defines a name for each of the 68 landmarks.

.. _building_blocks_landmarkface:

LandmarkFace
------------
:code:`pychubby.detect.LandmarkFace` is one of the most important classes that :code:`pychubby` uses.
To construct a :code:`LandmarkFace` one needs to provide

1. Image of the face
2. 68 landmark points

Rather than using the lower level constructor the user will mostly create instances through the class
method :code:`estimate` which detect the landmark points automatically.

Once instantiated, one can use actions (:code:`pychubby.actions.Action`) to generate a new (warped) 
:code:`LandmarkFace`. 


LandmarkFaces
-------------
:code:`pychubby.detect.LandmarkFaces` is a container holding multiple instances of :code:`LandmarkFace`. It
additionally provides functionality that allows for performing the :code:`Multiple` action on them.


Action
------
Action is a specific warping recipe that might depend on some parameters. Once instantiated, 
one can use their :code:`perform` method to warp a :code:`LandmarkFace`. To see already available
actions go to :ref:`gallery`  or read how to create your own actions :ref:`custom_actions`.

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



DisplacementField
-----------------
Displacement field represents a 2D to 2D transformations between two images. 
To instantiate a :code:`pychubby.base.DisplacementField` one can either use the standard 
constructor (:code:`delta_x`, :code:`delta_y` arrays).
Alternatively, one can use a factory method :code:`generate` that creates a :code:`DisplacemetField` based on
displacement of landmark points.
