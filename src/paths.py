import pathlib

"""Constants"""


ROOT_DIR = pathlib.Path(__file__).parent.parent
"""
The root directory of the project.
"""

DATA_DIR = ROOT_DIR / "data"
"""
The directory in which data is stored.
"""

OUT_DIR = ROOT_DIR / "out"
"""
The directory in which output is stored.
"""
