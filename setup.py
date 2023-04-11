from glob import glob

from setuptools import find_packages, setup

setup(
    name="gpt-mini",
    version="0.1",
    packages=find_packages(),
    url="",
    license="",
    author="",
    author_email="",
    description="GPT-mini libraries",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    data_files=[
        ("workflows", [x for x in glob("workflows/**", recursive=True) if "." in x])
    ],
    install_requires=[
        "pre-commit==3.2.0",
        "matplotlib==3.7.1",
    ],
    extras_require={
        "mac": [],
        "linux": [
            "tensorflow==2.12.0",
            "tensorflow-datasets==4.9.0",
        ],
    },
)
