.. _CLI:

CLI
===
:code:`pychubby` also offers a simple Command Line Interface that exposes some of the functionality of the
Python package.

Usage
-----
After installation of :code:`pychubby` an entry point :code:`pc` becomes available.

To see the basic information write :code:`pc --help`:

.. code-block:: bash

	Usage: pc [OPTIONS] COMMAND [ARGS]...

	  Automated face warping tool.

	Options:
	  --help  Show this message and exit.

	Commands:
	  list     List available actions.
	  perform  Take an action.


To perform actions one uses the :code:`perform` subcommand. :code:`pc perform --help`:

.. code-block:: bash

	Usage: pc perform [OPTIONS] COMMAND [ARGS]...

	  Take an action.

	Options:
	  --help  Show this message and exit.

	Commands:
	  Chubbify         Make a chubby face.
	  LinearTransform  Linear transformation.
	  OpenEyes         Open eyes.
	  ...
	  ...
	  ...


The syntax for all actions is identical. The **positional arguments** are

1. Input image path (required)
2. Output image path (not required)

If the output image path is not provided the resulting image is simply going to be plotted.

All the **options** correspond to the keyword arguments of the constructor of the respective action classes in :code:`pychubby.actions` module.

To give a specific example let us use the :code:`Smile` action. To get info on the parameters write
:code:`pc perform Smile --help`:

.. code-block:: bash

	Usage: pc perform Smile [OPTIONS] INP_IMG [OUT_IMG]

	  Make a smiling face.

	Options:
	  --scale FLOAT
	  --help         Show this message and exit.


In particular, one can then warp an image in the following fashion

.. code-block:: bash

	pc perform Smile --scale 0.3 img_cousin.jpg img_cousing_smiling.jpg




Limitations
-----------
The features that are unavailable via the CLI are the following:

1. :code:`AbsoluteMove`, :code:`Lambda` and :code:`Pipeline` actions
2. Different actions for different people
3. Lower level control

Specifically, if the user provides a photo with multiple faces the same action
will be performed on all of them.
