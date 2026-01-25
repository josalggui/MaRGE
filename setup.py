from setuptools import setup, find_packages

setup(
    name="marge-mri",
    version="0.0.12b2",
    author="José Miguel Algarín",
    author_email="josalggui@i3m.upv.es",
    packages=find_packages(),
    install_requires=[],
    description="MaRCoS Graphical Environment (MaRGE)",
    entry_points={
        "console_scripts": [
            "marge-mri=marge.main:MaRGE",
        ],
    },
)

