from setuptools import setup

setup(
	name = 'LogisticRegression',
	version = '1.0',
	description = 'Python Library for Supervised Learning Algorithm, Logistic Regression',
	author = 'Geetha Rangaswamaiah',
	author_email = 'rgeetha2010@gmail.com',
	packages = ['LogisticRegression'],
	install_requires = [
		'pandas',
		'numpy',
		'scipy'
		],
	classifiers = [
		'License :: MIT License',
		'Programming Language :: Python :: 3.0',
		'Topic :: Classification :: Logistic Regression'
		],
	include_package_data = True
)
