import setuptools

f = open("README.md", "r", encoding="utf-8")
long_description = f.read()
f.close()

setuptools.setup(
	name = "CIMAP",
	version = "0.0.10",
	author = "Gregorio Dotti",
	author_email = "gregorio.dotti@polito.it",
	description = "A Python package for muscle activation pattern analysis",
	install_requires = [
	'numpy>=1.23.2',
	'matplotlib>=3.5.3',
	'scipy>=0.17.1',
	'seaborn>=0.9.1'
	],
	packages = ["CIMAP"],
	classifiers=[
        "Programming Language :: Python :: 3 :: Only",
	"License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0', 

)
