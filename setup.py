from pathlib import Path
from setuptools import find_packages, setup

this_directory = Path(__file__).parent


long_description = (this_directory / ".pip_readme.rst").read_text()
requirements = (
    (this_directory / "requirements" / "requirements-core.txt").read_text().split("\n")
)


setup(
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    name="s2fft",
    version="0.0.1",
    url="https://github.com/astro-informatics/s2fft",
    author="Matthew A. Price, Jason D. McEwen & Contributors",
    license="GNU General Public License v3 (GPLv3)",
    python_requires=">=3.8",
    install_requires=requirements,
    description=(
        "Differentiable and accelerated spin-spherical harmonic transforms with JAX"
    ),
    long_description_content_type="text/x-rst",
    long_description=long_description,
    packages=find_packages(),
    include_package_data=True,
    pacakge_data={"s2fft": ["default-logging-config.yaml"]},
)
