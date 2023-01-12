import setuptools
import os
import re

try:
    # pip >=20
    from pip._internal.network.session import PipSession
    from pip._internal.req import parse_requirements
except ImportError:
    try:
        # 10.0.0 <= pip <= 19.3.1
        from pip._internal.download import PipSession
        from pip._internal.req import parse_requirements
    except ImportError:
        # pip <= 9.0.3
        from pip.download import PipSession
        from pip.req import parse_requirements
        
from setuptools.command.install import install

f = open("README.md", "r", encoding="utf-8")
long_description = f.read()
f.close()

from setuptools.command.install import install

class InstallReqs(install):
    def run(self):
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
        print(" ~~~~ Installing CIMAP ~~~~ ")
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
        os.system('pip install -r requirements.txt')
        install.run(self)



def resource(*args):
    return os.path.join(os.path.abspath(os.path.join(__file__, os.pardir)),
                        *args)
                        
                        


PACKAGES = ['CIMAP']

reqs = parse_requirements(resource('requirements.txt'), session=PipSession)

    
    
setuptools.setup(
	name = "CIMAP",

	version = "1.1.0",

	author = "Gregorio Dotti",
	author_email = "gregorio.dotti@polito.it",
	description = "A Python package for muscle activation pattern analysis",
	url = 'https://github.com/marcoghislieri/CIMAP',
	license='MIT',
	cmdclass={'install': InstallReqs},
	install_requires = [
	'numpy',
	'matplotlib',
	'scipy',
	'seaborn'
	],
	packages = PACKAGES,
	classifiers=[
        "Programming Language :: Python :: 3 :: Only",
	    "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8', 

)
