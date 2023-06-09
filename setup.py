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
        "mac": [
            "tensorflow-macos==2.9.2",
            "tensorflow-metal==0.5.1",
            "tensorflow-datasets==4.5.2",
            "torch==2.0.0",
            "transformers==4.27.4",
            "protobuf==3.19.6",
        ],
        "linux": [
            "tensorflow==2.12.0",
            "tensorflow-datasets==4.9.0",
            "torch==2.0.0",
            "transformers==4.27.4",
            "protobuf==3.20.3",
        ],
    },
)
