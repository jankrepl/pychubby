"""Definition of actions.

Note that for each action (class) the first line of the docstring
as well as the default parameters of the constructor are used by
the CLI.
"""

from abc import ABC, abstractmethod

import numpy as np
from skimage.transform import AffineTransform

from pychubby.base import DisplacementField
from pychubby.detect import LANDMARK_NAMES, LandmarkFace, LandmarkFaces
from pychubby.reference import DefaultRS


class Action(ABC):
    """General Action class to be subclassed."""

    @abstractmethod
    def perform(self, lf, **kwargs):
        """Perfom action on an instance of a LandmarkFace.

        Parameters
        ----------
        lf : LandmarkFace
            Instance of a ``LandmarkFace``.

        kwargs : dict
            Action specific parameters.

        Returns
        -------
        new_lf : LandmarkFace
            Instance of a ``LandmarkFace`` after a specified action was
            taken on the input `lf`.

        """

    @staticmethod
    def pts2inst(new_points, lf, **interpolation_kwargs):
        """Generate instance of LandmarkFace via interpolation.

        Parameters
        ----------
        new_points : np.ndarray
            Array of shape `(N, 2)` representing the x and y coordinates of the
            new landmark points.

        lf : LandmarkFace
            Instance of a ``LandmarkFace`` before taking any actions.

        interpolation_kwargs : dict
            Interpolation parameters passed onto scipy.

        Returns
        -------
        new_lf : LandmarkFace
            Instance of a ``LandmarkFace`` after taking an action.

        df : DisplacementField
            Displacement field representing per pixel displacements between the
            old and new image.

        """
        if not interpolation_kwargs:
            interpolation_kwargs = {"function": "linear"}

        df = DisplacementField.generate(
            lf.img.shape[:2],
            lf.points,
            new_points,
            anchor_edges=True,
            **interpolation_kwargs
        )

        new_img = df.warp(lf.img)

        return LandmarkFace(new_points, new_img), df


class AbsoluteMove(Action):
    """Absolute offsets of any landmark points.

    Parameters
    ----------
    x_shifts : dict or None
        Keys are integers from 0 to 67 representing a chosen landmark points. The
        values represent the shift in the x direction to be made. If a landmark
        not specified assumed shift is 0.

    y_shifts : dict or None
        Keys are integers from 0 to 67 representing a chosen landmark points. The
        values represent the shift in the y direction to be made. If a landmark
        not specified assumed shift is 0.

    """

    def __init__(self, x_shifts=None, y_shifts=None):
        """Construct."""
        self.x_shifts = x_shifts or {}
        self.y_shifts = y_shifts or {}

    def perform(self, lf):
        """Perform absolute move.

        Specified landmarks will be shifted in either the x or y direction.

        Parameters
        ----------
        lf : LandmarkFace
            Instance of a ``LandmarkFace``.

        Returns
        -------
        new_lf : LandmarkFace
            Instance of a ``LandmarkFace`` after taking the action.

        df : DisplacementField
            Displacement field representing the transformation between the old and
            new image.

        """
        offsets = np.zeros((68, 2))

        # x shifts
        for k, v in self.x_shifts.items():
            offsets[k, 0] = v
        # y shifts
        for k, v in self.y_shifts.items():
            offsets[k, 1] = v

        new_points = lf.points + offsets

        new_lf, df = self.pts2inst(new_points, lf)

        return new_lf, df


class Lambda(Action):
    """Custom action for specifying actions with angles and norms in a reference space.

    Parameters
    ----------
    scale : float
        Absolute norm of the maximum shift. All the remaining shifts are scaled linearly.

    specs : dict
        Dictionary where keyrs represent either the index or a name of the landmark.
        The values are tuples of two elements:
            1) Angle in degrees.
            2) Proportional shift. Only the relative size towards other landmarks matters.

    reference_space : None or ReferenceSpace
        Reference space to be used.

    """

    def __init__(self, scale, specs, reference_space=None):
        """Construct."""
        self.scale = scale
        self.specs = specs
        self.reference_space = reference_space or DefaultRS()

    def perform(self, lf):
        """Perform action.

        Parameters
        ----------
        lf : LandmarkFace
            Instance of a ``LandmarkFace`` before taking the action.

        Returns
        -------
        new_lf : LandmarkFace
            Instance of a ``LandmarkFace`` after taking the action.

        df : DisplacementField
            Displacement field representing the transformation between the old and new image.

        """
        self.reference_space.estimate(lf)
        ref_points = self.reference_space.inp2ref(lf.points)

        # Create entry for AbsoluteMove
        x_shifts = {}
        y_shifts = {}

        for k, (angle, prop) in self.specs.items():
            key = k if isinstance(k, int) else LANDMARK_NAMES[k]

            ref_shift = (
                np.array([[np.cos(np.radians(angle)), np.sin(np.radians(angle))]])
                * prop
                * self.scale
            )
            new_inp_point = self.reference_space.ref2inp(ref_points[key] + ref_shift)[0]
            shift = new_inp_point - lf.points[key]

            x_shifts[key] = shift[0]
            y_shifts[key] = shift[1]

        am = AbsoluteMove(x_shifts=x_shifts, y_shifts=y_shifts)

        return am.perform(lf)


class Chubbify(Action):
    """Make a chubby face.

    Parameters
    ----------
    scale : float
        Absolute shift size in the reference space.

    """

    def __init__(self, scale=0.2):
        """Construct."""
        self.scale = scale

    def perform(self, lf):
        """Perform an action.

        Parameters
        ----------
        lf : LandmarkFace
            Instance of a ``LandmarkFace``.

        """
        specs = {
            "LOWER_TEMPLE_L": (170, 0.4),
            "LOWER_TEMPLE_R": (10, 0.4),
            "UPPERMOST_CHEEK_L": (160, 1),
            "UPPERMOST_CHEEK_R": (20, 1),
            "UPPER_CHEEK_L": (150, 1),
            "UPPER_CHEEK_R": (30, 1),
            "LOWER_CHEEK_L": (140, 1),
            "LOWER_CHEEK_R": (40, 1),
            "LOWERMOST_CHEEK_L": (130, 0.8),
            "LOWERMOST_CHEEK_R": (50, 0.8),
            "CHIN_L": (120, 0.7),
            "CHIN_R": (60, 0.7),
            "CHIN": (90, 0.7),
        }

        return Lambda(self.scale, specs).perform(lf)


class LinearTransform(Action):
    """Linear transformation.

    Parameters
    ----------
    scale_x : float
        Scaling of the x axis.

    scale_y : float
        Scaling of the y axis.

    rotation : float
        Rotation in radians.

    shear : float
        Shear in radians.

    translation_x : float
        Translation in the x direction.

    translation_y : float
        Translation in the y direction.

    reference_space : None or pychubby.reference.ReferenceSpace
        Instace of the ``ReferenceSpace`` class.

    """

    def __init__(
        self,
        scale_x=1.,
        scale_y=1.,
        rotation=0.,
        shear=0.,
        translation_x=0.,
        translation_y=0.,
        reference_space=None,
    ):
        """Construct."""
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.rotation = rotation
        self.shear = shear
        self.translation_x = translation_x
        self.translation_y = translation_y
        self.reference_space = reference_space or DefaultRS()

    def perform(self, lf):
        """Perform action.

        Parameters
        ----------
        lf : LandmarkFace
            Instance of a ``LandmarkFace`` before taking the action.

        Returns
        -------
        new_lf : LandmarkFace
            Instance of a ``LandmarkFace`` after taking the action.

        df : DisplacementField
            Displacement field representing the transformation between the old and new image.

        """
        # estimate reference space
        self.reference_space.estimate(lf)

        # transform reference space landmarks
        ref_points = self.reference_space.inp2ref(lf.points)

        tform = AffineTransform(
            scale=(self.scale_x, self.scale_y),
            rotation=self.rotation,
            shear=self.shear,
            translation=(self.translation_x, self.translation_y),
        )
        tformed_ref_points = tform(ref_points)

        # ref2inp
        tformed_inp_points = self.reference_space.ref2inp(tformed_ref_points)

        x_shifts = {i: (tformed_inp_points[i] - lf[i])[0] for i in range(68)}
        y_shifts = {i: (tformed_inp_points[i] - lf[i])[1] for i in range(68)}

        return AbsoluteMove(x_shifts=x_shifts, y_shifts=y_shifts).perform(lf)


class Multiple(Action):
    """Applying actions to multiple faces.

    Parameters
    ----------
    per_face_action : list or Action
        If list then elements are instances of some actions (subclasses of ``Action``) that
        exactly match the order of ``LandmarkFace`` instances within the ``LandmarkFaces``
        instance. It is also posible to use None for no action. If ``Action`` then the
        same action will be performed on each available ``LandmarkFace``.

    """

    def __init__(self, per_face_action):
        """Construct."""
        if isinstance(per_face_action, list):
            if not all([isinstance(a, Action) or a is None for a in per_face_action]):
                raise TypeError("All elements of per_face_action need to be actions.")

            self.per_face_action = per_face_action

        elif isinstance(per_face_action, Action) or per_face_action is None:
            self.per_face_action = [per_face_action]

        else:
            raise TypeError(
                "per_face_action needs to be an action or a list of actions"
            )

    def perform(self, lfs):
        """Perform actions on multiple faces.

        Parameters
        ----------
        lfs : LandmarkFaces
            Instance of ``LandmarkFaces``.

        Returns
        -------
        new_lfs : LandmarkFaces
            Instance of a ``LandmarkFaces`` after taking the action on each face.

        df : DisplacementField
            Displacement field representing the transformation between the old and new image.

        """
        if isinstance(lfs, LandmarkFace):
            lfs = LandmarkFaces(lfs)

        n_actions = len(self.per_face_action)
        n_faces = len(lfs)

        if n_actions not in {1, n_faces}:
            raise ValueError(
                "Number of actions ({}) is different from number of faces({})".format(
                    n_actions, n_faces
                )
            )

        lf_list_new = []
        for lf, a in zip(
            lfs,
            self.per_face_action if n_actions != 1 else n_faces * self.per_face_action,
        ):
            lf_new, _ = a.perform(lf) if a is not None else (lf, None)
            lf_list_new.append(lf_new)

        # Overall displacement
        img = lfs[0].img
        shape = img.shape[:2]
        old_points = np.vstack([lf.points for lf in lfs])
        new_points = np.vstack([lf.points for lf in lf_list_new])

        df = DisplacementField.generate(
            shape, old_points, new_points, anchor_corners=True, function="linear"
        )

        # Make sure same images
        img_final = df.warp(img)
        lfs_new = LandmarkFaces(
            *[LandmarkFace(lf.points, img_final) for lf in lf_list_new]
        )

        return lfs_new, df


class OpenEyes(Action):
    """Open eyes.

    Parameters
    ----------
    scale : float
        Absolute shift size in the reference space.

    """

    def __init__(self, scale=0.1):
        """Construct."""
        self.scale = scale

    def perform(self, lf):
        """Perform action.

        Parameters
        ----------
        lf : LandmarkFace
            Instance of a ``LandmarkFace`` before taking the action.

        Returns
        -------
        new_lf : LandmarkFace
            Instance of a ``LandmarkFace`` after taking the action.

        df : DisplacementField
            Displacement field representing the transformation between the old and new image.

        """
        specs = {
            "INNER_EYE_LID_R": (-100, 0.8),
            "OUTER_EYE_LID_R": (-80, 1),
            "INNER_EYE_BOTTOM_R": (100, 0.5),
            "OUTER_EYE_BOTTOM_R": (80, 0.5),
            "INNERMOST_EYEBROW_R": (-100, 1),
            "INNER_EYEBROW_R": (-100, 1),
            "MIDDLE_EYEBROW_R": (-100, 1),
            "OUTER_EYEBROW_R": (-100, 1),
            "OUTERMOST_EYEBROW_R": (-100, 1),
            "INNER_EYE_LID_L": (-80, 0.8),
            "OUTER_EYE_LID_L": (-100, 1),
            "INNER_EYE_BOTTOM_L": (80, 0.5),
            "OUTER_EYE_BOTTOM_L": (10, 0.5),
            "INNERMOST_EYEBROW_L": (-80, 1),
            "INNER_EYEBROW_L": (-80, 1),
            "MIDDLE_EYEBROW_L": (-80, 1),
            "OUTER_EYEBROW_L": (-80, 1),
            "OUTERMOST_EYEBROW_L": (-80, 1),
        }
        return Lambda(self.scale, specs=specs).perform(lf)


class Pipeline(Action):
    """Pipe multiple actions together.

    Parameters
    ----------
    steps : list
        List of different actions that are going to be performed in the given order.

    """

    def __init__(self, steps):
        """Construct."""
        self.steps = steps

    def perform(self, lf):
        """Perform action.

        Parameters
        ----------
        lf : LandmarkFace
            Instance of a ``LandmarkFace`` before taking the action.

        Returns
        -------
        new_lf : LandmarkFace
            Instance of a ``LandmarkFace`` after taking the action.

        df : DisplacementField
            Displacement field representing the transformation between the old and new image.

        """
        df_list = []
        lf_composed = lf

        for a in self.steps:
            lf_composed, df_temp = a.perform(lf_composed)
            df_list.append(df_temp)

        df_list = df_list[::-1]

        df_composed = df_list[0]
        for df_temp in df_list[1:]:
            df_composed = df_composed(df_temp)

        return lf_composed, df_composed


class RaiseEyebrow(Action):
    """Raise an eyebrow.

    Parameters
    ----------
    scale : float
        Absolute shift size in the reference space.

    side : str, {'left', 'right', 'both'}
        Which eyebrow to raise.

    """

    def __init__(self, scale=0.1, side="both"):
        """Construct."""
        self.scale = scale
        self.side = side

        if self.side not in {"left", "right", "both"}:
            raise ValueError(
                "Allowed side options are 'left', 'right' and 'both'.".format(self.side)
            )

    def perform(self, lf):
        """Perform action.

        Parameters
        ----------
        lf : LandmarkFace
            Instance of a ``LandmarkFace`` before taking the action.

        Returns
        -------
        new_lf : LandmarkFace
            Instance of a ``LandmarkFace`` after taking the action.

        df : DisplacementField
            Displacement field representing the transformation between the old and new image.

        """
        sides = []
        if self.side in {"both", "left"}:
            sides.append("L")

        if self.side in {"both", "right"}:
            sides.append("R")

        specs = {}
        for side in sides:
            specs.update(
                {
                    "OUTERMOST_EYEBROW_{}".format(side): (-90, 1),
                    "OUTER_EYEBROW_{}".format(side): (-90, 0.7),
                    "MIDDLE_EYEBROW_{}".format(side): (-90, 0.4),
                    "INNER_EYEBROW_{}".format(side): (-90, 0.2),
                    "INNERMOST_EYEBROW_{}".format(side): (-90, 0.1),
                }
            )

        return Lambda(self.scale, specs).perform(lf)


class Smile(Action):
    """Make a smiling face.

    Parameters
    ----------
    scale : float
        Absolute shift size in the reference space.

    """

    def __init__(self, scale=0.1):
        """Construct."""
        self.scale = scale

    def perform(self, lf):
        """Perform action.

        Parameters
        ----------
        lf : LandmarkFace
            Instance of a ``LandmarkFace`` before taking the action.

        Returns
        -------
        new_lf : LandmarkFace
            Instance of a ``LandmarkFace`` after taking the action.

        df : DisplacementField
            Displacement field representing the transformation between the old and new image.

        """
        specs = {
            "OUTSIDE_MOUTH_CORNER_L": (-110, 1),
            "OUTSIDE_MOUTH_CORNER_R": (-70, 1),
            "INSIDE_MOUTH_CORNER_L": (-110, 0.8),
            "INSIDE_MOUTH_CORNER_R": (-70, 0.8),
            "OUTER_OUTSIDE_UPPER_LIP_L": (-100, 0.3),
            "OUTER_OUTSIDE_UPPER_LIP_R": (-80, 0.3),
        }

        return Lambda(self.scale, specs).perform(lf)


class StretchNostrils(Action):
    """Stratch nostrils.

    Parameters
    ----------
    scale : float
        Absolute shift size in the reference space.

    """

    def __init__(self, scale=0.1):
        """Construct."""
        self.scale = scale

    def perform(self, lf):
        """Perform action.

        Parameters
        ----------
        lf : LandmarkFace
            Instance of a ``LandmarkFace`` before taking the action.

        Returns
        -------
        new_lf : LandmarkFace
            Instance of a ``LandmarkFace`` after taking the action.

        df : DisplacementField
            Displacement field representing the transformation between the old and new image.

        """
        specs = {"OUTER_NOSTRIL_L": (-135, 1), "OUTER_NOSTRIL_R": (-45, 1)}

        return Lambda(self.scale, specs).perform(lf)
