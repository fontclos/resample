from setuptools import setup

setup(
    name="resample",
    version="0.1",
    py_modules=["resample"],
    install_requires=[
        "KDEpy",
        "scikit-learn",
        "numpy",
    ]
)
