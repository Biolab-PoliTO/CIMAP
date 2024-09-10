import setuptools
import os
from setuptools.command.install import install

class InstallReqs(install):
    """Custom command to install requirements from requirements.txt."""
    def run(self):
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
        print(" ~~~~ Installing CIMAP ~~~~ ")
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
        os.system('pip install -r requirements.txt')
        # Call the parent class's run method correctly
        install.run(self)

# Read the contents of your README file for the long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

PACKAGES = ['CIMAP']

setuptools.setup(
    name="CIMAP",
    version="1.1.1",
    author="Gregorio Dotti",
    author_email="gregorio.dotti@polito.it",
    description="A Python package for muscle activation pattern analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/marcoghislieri/CIMAP',
    license='MIT',
    cmdclass={'install': InstallReqs},
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'seaborn',
    ],
    packages=PACKAGES,
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)