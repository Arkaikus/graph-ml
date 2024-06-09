from setuptools import find_packages, setup

setup(
    name="earthquakes",
    version="1.0.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "quakes = main:main",
        ],
    },
)
