Pipelines
=========
Rather than applying a single action at a time :code:`pychubby` enables piping multiple actions together.
To achieve this one can use the metaaction :code:`Pipeline`.

Let us again assume that we start with an image with a single face in it.

.. image:: https://i.imgur.com/3yAhFzi.jpg
  :width: 400
  :alt: Original image
  :align: center


Let's try to make the person smile but also close  her eyes slightly.

.. code-block:: python
	
	import matplotlib.pyplot as plt

	from pychubby.actions import OpenEyes, Pipeline, Smile
	from pychubby.detect import LandmarkFace

	img = plt.imread("path/to/the/image")
	lf = LandmarkFace.estimate(img)

	a_s = Smile(0.1)
	a_e = OpenEyes(-0.03)
	a = Pipeline([a_s, a_e])

	new_lf, df = a.perform(lf)
	new_lf.plot(show_landmarks=False)



.. image:: https://i.imgur.com/E1BdvBq.jpg
  :width: 400
  :alt: Warped image
  :align: center

To create an animation we can use the `visualize` module.

.. code-block:: python
	
	from pychubby.visualize import create_animation

	ani = create_animation(df, img)

.. image:: https://i.imgur.com/PlCqUZr.gif
  :width: 400
  :alt: Animation
  :align: center
