import setuptools

f = open("README.md", "r", encoding="utf-8")
long_description = f.read()
f.close()

setuptools.setup(
	name = "CIMAP",

	version = "0.1.0",

	author = "Gregorio Dotti",
	author_email = "gregorio.dotti@polito.it",
	description = "A Python package for muscle activation pattern analysis",
	install_requires = [
	'numpy',
	'matplotlib',
	'scipy',
	'seaborn'
	],
	packages = ["CIMAP"],
	classifiers=[
        "Programming Language :: Python :: 3 :: Only",
	"License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8', 

)
