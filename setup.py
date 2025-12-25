from setuptools import setup, Extension
import pathlib

HERE = pathlib.Path(__file__).parent

percy_extension = Extension(
    name="percy.per",                # the actual import module name
    sources=[str(HERE / "percy" / "per.c")],
    extra_compile_args=["-O3"],
)

setup(
    name="percy-rl",
    version="0.1.0",
    description="Prioritized Experience Replay in C with Python bindings",
    long_description=(HERE / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Redzhep Mehmedov",
    author_email="redzhep12@gmail.com",
    ext_modules=[percy_extension],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: C",
    ],
    python_requires=">=3.6",
)