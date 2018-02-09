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

## Notes ##
Since the CoreRank algorithm need get core number for each vertex considering the weight of each edge and networkX itself doesn't take it into account. The networkX source code need to be modified. 

Find `$KERAS_INSTALLATION_HOME/algorithms/core.py`, replace function `core_number` as below:
```python
def core_number(G, weight=None):
    """Return the core number for each vertex.

    A k-core is a maximal subgraph that contains nodes of degree k or more.

    The core number of a node is the largest value k of a k-core containing
    that node.

    Parameters
    ----------
    G : NetworkX graph
       A graph or directed graph

    Returns
    -------
    core_number : dictionary
       A dictionary keyed by node to the core number.

    Raises
    ------
    NetworkXError
        The k-core is not defined for graphs with self loops or parallel edges.

    Notes
    -----
    Not implemented for graphs with parallel edges or self loops.

    For directed graphs the node degree is defined to be the
    in-degree + out-degree.

    References
    ----------
    .. [1] An O(m) Algorithm for Cores Decomposition of Networks
       Vladimir Batagelj and Matjaz Zaversnik, 2003.
       http://arxiv.org/abs/cs.DS/0310049
    """
    if G.is_multigraph():
        raise nx.NetworkXError(
                'MultiGraph and MultiDiGraph types not supported.')

    if G.number_of_selfloops()>0:
        raise nx.NetworkXError(
                'Input graph has self loops; the core number is not defined.',
                'Consider using G.remove_edges_from(G.selfloop_edges()).')

    if G.is_directed():
        import itertools
        def neighbors(v):
            return itertools.chain.from_iterable([G.predecessors_iter(v),
                                                  G.successors_iter(v)])
    else:
        neighbors=G.neighbors_iter
    # modifed start
    degrees=G.degree(weight=weight)
    if weight:
        for k in degrees:
            degrees[k] = int(degrees[k])
    # modifed end

    # sort nodes by degree
    nodes=sorted(degrees,key=degrees.get)
    bin_boundaries=[0]
    curr_degree=0
    for i,v in enumerate(nodes):
        if degrees[v]>curr_degree:
            bin_boundaries.extend([i]*(degrees[v]-curr_degree))
            curr_degree=degrees[v]
    node_pos = dict((v,pos) for pos,v in enumerate(nodes))
    # initial guesses for core is degree
    core=degrees
    nbrs=dict((v,set(neighbors(v))) for v in G)
    for v in nodes:
        for u in nbrs[v]:
            if core[u] > core[v]:
                nbrs[u].remove(v)
                pos=node_pos[u]
                bin_start=bin_boundaries[core[u]]
                node_pos[u]=bin_start
                node_pos[nodes[bin_start]]=pos
                nodes[bin_start],nodes[pos]=nodes[pos],nodes[bin_start]
                bin_boundaries[core[u]]+=1
                core[u]-=1
    return core
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