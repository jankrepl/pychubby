.. _custom_actions:

==============
Custom Actions
==============

:code:`pychubby` makes it very easy to add custom actions. There are 3 main ingredients:

1. Each action needs to be a subclass of  :code:`pychubby.actions.Action` 
2. All parameters of the action are specified via the constructor (:code:`__init__`)
3. The method :code:`perform` needs to be implemented such that

	- It inputs an instance of the :code:`pychubby.detect.LandmarkFace`
	- It returns a new instance of :code:`pychubby.detect.LandmarkFace` and :code:`pychubby.base.DisplacementField` representing the pixel by pixel transformation from the new image to the old one.


Clearly the main workhorse is the 3rd step. In order to avoid dealing with lower level details
a good start is to use the utility action :code:`Lambda`.

.. _custom_actions_lambda:

------
Lambda
------
The simplest way how to implement a new action is to use the :code:`Lambda` action. Before explaining
the action itself the reader is encouraged to review the :code:`DefaultRS` reference space in 
:ref:`building_blocks_reference_space` which is by default used by :code:`Lambda`. 



.. image:: https://i.imgur.com/dLcFQNI.gif 
   :width: 500
   :align: center

The lambda action works purely in the reference space and expects the following input:

- **scale** - float representing the absolute size (norm) of the largest displacement in the reference space (this would be the chin displacement in the figure)
- **specs** - dictionary where keys are landmarks (either name or number) and the values are tuples (angle, relative size)


That means that the user simply specifies for landmark of interest what is the displacement 
angle and relative size with respect to all other displacements through the :code:`specs` dictionary. 
After that the :code:`scale` parameter controls the absolute size of the biggest displacement while the other
displacements are scaled linearly based on the provided relative sizes. 

See below an example that replicates the figure:

.. code-block:: python
	
	from pychubby.actions import Action, Lambda

	class CustomAction(Action):

	    def __init__(self, scale=0.3):
		self.scale = scale

	    def perform(self, lf):
		a_l = Lambda(scale=self.scale,
			     specs={'CHIN': (90, 2),
				    'CHIN_L': (110, 1),
				    'CHIN_R': (70, 1),
				    'OUTER_NOSTRIL_L': (-135, 1),
				    'OUTER_NOSTRIL_R': (-45, 1)
				   }
			    )
		
		return a_l.perform(lf)
    


.. image:: https://i.imgur.com/VqmXtzU.gif
   :width: 300
   :align: center
