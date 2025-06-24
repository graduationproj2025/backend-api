from setuptools import setup

setup(
    name="backend-api",
    version="1.0.0",  # تأكد من تطابق هذا مع __init__.py
    packages=["backend-api"],
    install_requires=open("requirements.txt").read().splitlines(),
)