class Graph:

    def __init__(self, M):
        """Graph: a generic undirected graph

        Args:
            M (list): an adjacency matrix representation of the graph,
                      where M[i][j] represents the flow capacity on E(i,j).
        """

        self.M = M

    def _find_path(self, source, destination, flows):
        """Conducts a BFS and tries finds a traversable path
           from source to destination, considering flows and capacities.

        Args:
            source (int): index of source in self.M
            destination (int): index of destination in self.M
            flows (list): an adjacency matrix of the same dimensions as self.M,
                          but listing the current flow in each edge.

        Returns:
            list: a list of indices containing a path if found, or None if not.
        """
        
        # BFS implementation follows, adapted to return path
        discovered = [False] * len(self.M)
        queue = [[source]]
        while queue:
            this_path = queue.pop()
            this_node = this_path[-1]
            if this_node == destination:
                return this_path
            for neighbor, capacity in enumerate(self.M[this_node]):
                if (not discovered[neighbor] 
                        and flows[this_node][neighbor] < capacity):
                    new_path = list(this_path)
                    new_path.append(neighbor)
                    discovered[neighbor] = True
                    queue.insert(0, new_path)

    def max_flow(self, sources, sinks):
        """Ford-Fulkerson algorithm for max flow.

        Args:
            sources (list): list of indices of sources
            sinks (list): list of indices of sinks

        Returns:
            int: The max flow found to the sinks provided.
        """

        """If there are multiple sources and/or sinks,
        we must convert the graph to one with a single source
        and sink. We do this by creating a new source with
        infinite flow capacity to the sources, and/or a new sink
        to which the sinks flow with infinite capacity."""

        if len(sources) > 1 or len(sinks) > 1:
            # create an ad-hoc new graph
            from copy import deepcopy
            new_g = Graph(deepcopy(self.M))
            
            # add 'supersink' and adjust matrix accordingly
            if len(sinks) > 1:
                for i, row in enumerate(new_g.M):
                    if i in sinks:
                        row.append(float('inf'))
                    else:
                        row.append(0)

                new_g.M.append([0 for _ in range(len(new_g.M) + 1)])

            # add 'supersource' and adjust matrix accordingly
            if len(sources) > 1:
                supersource = [0]
                supersource += [
                    (float('inf') if i in sources else 0)
                    for i in range(len(new_g.M))
                    ]
                new_g.M.insert(0, supersource)
                for row in new_g.M[1:]:
                    row.insert(0, 0)

            # we now return the max flow from a single source to a single sink
            return new_g.max_flow([0], [len(new_g.M) - 1])

        # if we are here, we should have a single source and sink
        source = sources[0]
        sink = sinks[0]

        # Ford-Fulkerson starts here
        flows = [[0 for _ in range(len(row))] for row in self.M]
        while True:
            path = self._find_path(source, sink, flows)
            if path is None:
                break
            max_allowable_flow = min(
                self.M[u][v] - flows[u][v]
                for u, v in zip(path[:-1], path[1:])
                )
            for u, v in zip(path[:-1], path[1:]):
                flows[u][v] += max_allowable_flow
                flows[v][u] -= max_allowable_flow

        return sum(row[sink] for row in flows)


def solution(entrances, exits, path):
    return Graph(path).max_flow(entrances, exits)
