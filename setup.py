from setuptools import find_packages, setup


def get_requirements(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith(("#", "-"))]


setup(
    name="student-performance-assistant",
    version="1.0.0",
    author="rishi",
    author_email="rishinavroop22@gmail.com",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.10",
)
