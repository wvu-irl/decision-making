from setuptools import setup

setup(
    name="decision_making",
    version="0.1.0",
    install_requires=["irl-gym", "gym", "pygame"],
)


# Set config params
import os

current = os.path.dirname(os.path.realpath(__file__))
from decision_making.config.core.set_default_config import set_defaults

set_defaults(current + "/config/core/")