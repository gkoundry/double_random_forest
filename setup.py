import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

ext_modules = [
    Extension(
        "double_random_forest.core",
        sources=[
            "double_random_forest/core.cpp",
            "double_random_forest/utils.cpp",
            "double_random_forest/tree.cpp",
            "double_random_forest/criterion.cpp",
        ],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=[
            "-Werror",  # Treat warnings as errors (GCC/Clang)
            "-Wall",  # Enable all warnings
            "-Wextra",  # Enable extra warnings
        ]
        if sys.platform != "win32"
        else [
            "/WX",  # Treat warnings as errors (MSVC)
        ],
        language="c++",
    ),
]

setup(
    name="double_random_forest",
    version="0.1",
    author="Glen Koundry",
    author_email="gkoundry@gmail.com",
    description="A Python implementation of the Double Random Forest algorithm",
    packages=["double_random_forest"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    install_requires=[
        "pybind11",
    ],
)
