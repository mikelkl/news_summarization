from setuptools import setup, find_packages

setup(
    name='newssum',
    version="0.1.0",
    author="Mike Liu",
    author_email="mikelkl@foxmail.com",
    url="https://github.com/mikelkl/news_summarization",
    keywords=[
        "data mining",
        "automatic summarization",
        "NLP",
        "natural language processing",
    ],
    install_requires=[
        "gensim >= 2.3.0",
        "matplotlib >= 2.1.1",
        "networkx == 1.11",
        "nltk >= 3.2.4",
        "numpy >= 1.13.1",
        "pandas >= 0.20.3",
        "pathos >= 0.2.1",
        "prettytable >= 0.7.2",
        "requests >= 2.18.4",
        "rouge >= 0.2.1",
        "scikit_learn >= 0.19.0",
        "sumy >= 0.7.0",
    ],
    packages=find_packages()
)