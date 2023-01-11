import pathlib
#from setuptools import find_packages, setup
from distutils.core import setup
import py2exe
# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="bbo-acm-traingui",
    version="0.11.7",
    description="GUI to label frames for training of ACM-dlcdetect",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/bbo-lab/acm-traingui",
    author="BBO-lab @ caesar",
    author_email="kay-michael.voit@caesar.de",
    license="BSD",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=['ACMtraingui'],
    include_package_data=True,
    install_requires=["numpy"],
    options = {'py2exe': 
        {
            "includes": [
                "sip",
                "PyQt5",
                "PyQt5.QtCore",
                "PyQt5.QtGui",
                "PyQt5.QtWidgets",
                "numpy",
                "matplotlib.backends.backend_qt5agg",
                "scipy",
                "scipy.sparse.csgraph._validation",
                "scipy.special._ufuncs_cxx",
            ]
        }},
    console = ["traingui.py"],
)
