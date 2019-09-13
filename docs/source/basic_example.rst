Basic Example
=============
To illustrate the simplest use case let us assume that we start with a photo with
a single face in it.

.. image:: https://i.imgur.com/3yAhFzi.jpg
  :width: 400
  :alt: Original image
  :align: center


:code:`pychubby` implements a class :code:`LandmarkFace` which stores  all relevant data that enable face warping.
Namely it is the image itself and 68 landmark points. To instantiate a :code:`LandmarkFace` one needs 
to use a utility class method :code:`estimate`.


.. code-block:: python

	import matplotlib.pyplot as plt
	from pychubby.detect import LandmarkFace

	img = plt.imread("path/to/the/image") 
	lf = LandmarkFace.estimate(img)
	lf.plot()

.. image:: https://i.imgur.com/y4AL171.png
  :width: 400
  :alt: Face with landmarks
  :align: center

Note that it might be necessary to upsample the image before the estimation. For convenience
the :code:`estimate` method has an optional parameter :code:`n_upsamples`.

Once the landmark points are estimated we can move on with performing actions on the face.
Let's try to make the person smile:

.. code-block:: python

	from pychubby.actions import Smile

	a = Smile(scale=0.2)
	new_lf, df = a.perform(lf)  # lf defined above
	new_lf.plot(show_landmarks=False)

.. image:: https://i.imgur.com/RytGu0t.png
  :width: 400
  :alt: Smiling face
  :align: center

There are 2 important things to note. Firstly the :code:`new_lf` now contains both the warped version
of the original image as well as the transformed landmark points. Secondly, the :code:`perform`
method also returns a :code:`df` which is an instance of :code:`pychubby.base.DisplacementField` and
represents the pixel by pixel transformation between the old and the new (smiling) image.

To see all currently available actions go to :ref:`gallery`.

To create an animation of the action we can use the :code:`visualize` module.

.. code-block:: python

	from pychubby.visualize import create_animation

	ani = create_animation(df, img) # the displacement field and the original image

.. image:: https://i.imgur.com/jB6Vlnc.gif 
   :width: 400
   :align: center
