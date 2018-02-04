# short-paths

This is an implementation of the algorithm to find short s-t paths in a graph
given by Eppstein in [this paper](http://citeseer.ist.psu.edu/viewdoc/download;jsessionid=7BB56B2ABC7C9113C121413A62AF3974?doi=10.1.1.30.3705&rep=rep1&type=pdf).
Specifically, it constructs the path graph as described in section 2,
then does a Dijkstra-esque walk from the root of the path graph to generate
paths in increasing order of length.

Because we use a basic implementation of Dijkstra's algorithm to generate the
shortest path tree, the preprocessing step is O(m log n).
Furthermore, since we are not using any of the sophisticated
heap selection algorithms and we are returning paths as full lists of edges rather
than implicitly, the time to generate the i'th path is O(log i + e) where e is the
number of edges in that path.
