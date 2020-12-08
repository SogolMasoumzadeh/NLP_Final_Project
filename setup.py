import setuptools
_packages = setuptools.find_packages()

requirements =[
    "nltk",
    "sklearn",
    "tqdm",
    "sentence_transformers",
    "pandas",
    "tensorflow",
    "matplotlib",
    "spacy"
    # python -m spacy download en_core_web_sm
]

with open ("README.md", "r") as f:
    long_desc = f.read()

setuptools.setup(
    name="Final_project",
    version="0.1.0",
    description="TODO",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    packages=_packages,
    python_requires=">=3.6, <4",
    install_requires=requirements,
)