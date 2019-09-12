"""Command line interface."""

import inspect

import click
import matplotlib.pyplot as plt

import pychubby.actions
from pychubby.detect import LandmarkFace

EXCLUDED_ACTIONS = ["Action", "AbsoluteMove", "Lambda", "Multiple", "Pipeline"]
ALL_ACTIONS = [
    m[0]
    for m in inspect.getmembers(pychubby.actions, inspect.isclass)
    if m[1].__module__ == "pychubby.actions" and m[0] not in EXCLUDED_ACTIONS
]


@click.group()
def cli():
    """Automated face warping tool."""
    pass


@cli.group()
def perform():
    """Take an action."""
    pass


class ActionFactory:
    """Utility for defining subcommands of the perform command dynamically.

    The goal is to have a separate CLI command for each action. To
    achieve this we use the specific structure of the `pychubby.actions`
    module. Namely we use the fact that each class corresponds to an
    action.

    Parameters
    ----------
    name : str
        Name of the action. The subcommand will be named the same.

    doc : str
        First line of the class docstring. Will be used for the
        help of the subcommand.

    Attributes
    ----------
    kwargs : dict
        All parameters (keys) of a given action together with their defaults (values).

    """

    def __init__(self, name, doc):
        """Construct."""
        self.name = name
        self.doc = doc

        try:
            spec = inspect.getargspec(getattr(pychubby.actions, self.name).__init__)
            self.kwargs = dict(zip(spec.args[-len(spec.defaults):], spec.defaults))
        except Exception:
            self.kwargs = {}

    def generate(self):
        """Define a subsubcommand."""
        operators = [
            perform.command(name=self.name, help=self.doc),
            click.argument("inp_img", type=click.Path(exists=True)),
            click.argument("out_img", type=click.Path(), required=False),
        ]

        operators += [
            click.option("--{}".format(k), default=v) for k, v in self.kwargs.items()
        ]

        def f(*args, **kwargs):
            """Perform an action."""
            inp_img = kwargs.pop("inp_img")
            out_img = kwargs.pop("out_img")

            img = plt.imread(str(inp_img))
            lf = LandmarkFace.estimate(img)
            cls = getattr(pychubby.actions, self.name)
            a = pychubby.actions.Multiple(cls(**kwargs))

            new_lf, df = a.perform(lf)

            if out_img is not None:
                plt.imsave(str(out_img), new_lf[0].img)
            else:
                new_lf.plot(show_landmarks=False, show_numbers=False)

        for op in operators[::-1]:
            f = op(f)

        return f


for action in ALL_ACTIONS:
    doc = getattr(pychubby.actions, action).__doc__.split("\n")[0]
    ActionFactory(action, doc).generate()


@cli.command()
def list(*args, **kwargs):
    """List available actions."""
    print("\n".join(ALL_ACTIONS))
