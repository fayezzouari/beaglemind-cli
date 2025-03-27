from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='beaglemind-cli',  # Replace with your actual package name
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'beaglemind=beaglemind.main:main',  # This maps the CLI command to the main function in main.py
        ],
    },
    author='Fayez Zouari',
    author_email='fayez.zouari@insat.ucar.tn',
    description='BeagleBoard AI CLI Assistant',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/fayezzouari/beaglemind-cli',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Adjust based on your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',  # Specify your Python version requirement
)