# Automatic text summarizer for news

Simple library for extracting summary from [Deepmind news dataset](https://cs.nyu.edu/~kcho/DMQA/) or plain texts. The package also contains simple evaluation framework for text summaries. Inspired by:

- **CoreRank** - [Combining Graph Degeneracy and Submodularity for Unsupervised Extractive Summarization](http://www.aclweb.org/anthology/W17-4507)
- **GoWvis** - [GoWvis: a web application for Graph-of-Words-based text visualization and summarization](http://www.aclweb.org/anthology/P16-4026)
- **RASR** - [A Redundancy-Aware Sentence Regression Framework for Extractive Summarization](http://www.aclweb.org/anthology/C16-1004)
- **Kam-Fai** Supervised method - [Extractive Summarization Using Supervised and Semi-supervised Learning](http://www.aclweb.org/anthology/C08-1124)
- **InfoFilter** - [Detecting (Un)Important Content for Single-Document News Summarization](http://aclweb.org/anthology/E17-2112)

## Installation ##
```python
python setup.py install
```

## Complete Project Structure ##
```
├───.idea
├───build
├───data               <- The original, immutable data dump.
├───dist
├───external
├───figures            <- Figures saved by notebooks and scripts.
├───newssum            <- Python package with source code.
│   ├───evaluation
│   ├───feature_extraction
│   ├───models
│   ├───parsers
│   ├───summarizers
├───newssum.egg-info
├───notebooks
├───output             <- Processed data, models, logs, etc.
├───tests              <- Tests for Python package.
├── README.md          <- README with info of the project.
├── server.py          <- Simple server for online demo.
└── setup.py           <- Install and distribute module.
```

## Usage ##
```python
from newssum.parsers import PlaintextParser
from newssum.summarizers import CoreRank

TEXT = "Thomas appeared in 15 games (14 starts) for Cleveland this season, averaging 14.7 points, 4.5 assists and 2.1 rebounds in 27.1 minutes. The two-time NBA All-Star (2015-17) owns career averages of 19.0 points (.441 FG%), 5.1 assists, 2.6 rebounds and 1.0 steals in 456 career games (323 starts). In 2016-17, Thomas earned All-NBA Second Team honors when he averaged a career-high 28.9 points (.463 FG%) per game."

if __name__ == "__main__":
    parser = PlaintextParser(TEXT)
    cr_summarizer = CoreRank(parser)
    summary = cr_summarizer.get_best_sents(w_threshold=25)
    print(summary)
```