# Automatic text summarizer for news

Simple library for extracting summary from [Deepmind news dataset](https://cs.nyu.edu/~kcho/DMQA/) or plain texts. The package also contains simple evaluation framework for text summaries. Inspired by:

- **CoreRank** - [Combining Graph Degeneracy and Submodularity for Unsupervised Extractive Summarization](http://www.aclweb.org/anthology/W17-4507)
- **GoWvis** - [GoWvis: a web application for Graph-of-Words-based text visualization and summarization](http://www.aclweb.org/anthology/P16-4026)
- **RASR** - [A Redundancy-Aware Sentence Regression Framework for Extractive Summarization](http://www.aclweb.org/anthology/C16-1004)
- **Kam-Fai** Supervised method - [Extractive Summarization Using Supervised and Semi-supervised Learning](http://www.aclweb.org/anthology/C08-1124)
- **InfoFilter** - [Detecting (Un)Important Content for Single-Document News Summarization](http://aclweb.org/anthology/E17-2112)

## Usage ##
```python
from newssum.parsers import PlaintextParser
from newssum.summarizers import CoreRank

TEXT = "When North Korea captured the American surveillance ship USS Pueblo, it sparked an 11-month crisis that threatened to worsen already high Cold War tensions in the region."

if __name__ == "__main__":
    parser = PlaintextParser(TEXT)
    cr_summarizer = CoreRank(parser)
    summary = cr_summarizer.get_best_sents()
    print(summary)
```