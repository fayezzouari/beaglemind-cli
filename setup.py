from setuptools import setup, find_packages
import os

# Read the contents of your requirements.txt file
with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as f:
    required = f.read().splitlines()

setup(
    name='beaglemind-cli',
    version='0.1.0',
    description='CLI tool for the BeagleMind agent',
    author='Fayez Zouari',
    author_email='fayez.zouari@insat.ucar.tn',
    packages=find_packages(),
    install_requires=required,
    entry_points={
        'console_scripts': [
            'beaglemind=beaglemind.main:main',
        ],
    },
)
