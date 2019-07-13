"""Command line interface."""

import click


@click.group()
def cli():
    """Make faces chubby again."""
    pass


@cli.command()
def convert():
    """Convert a normal face to a chubby face."""
    pass
