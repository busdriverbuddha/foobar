"""
"There's coffee in that nebula."
                -- Capt. Kathryn Janeway

This problem can be modeled as a tile-placement,
edge-matching game.

There are 16 possible 2x2 tiles, 4 of which
generate a True square and 12 of which generate
a False square. We write their binary equivalents.
For instance, the tiles that generate a True square
are

00 00 01 10
01 10 00 00

Reading left to right, top-down, we can number them as
1, 2, 4, and 8, respectively. We can take their edges as
each 2-bit combination along a direction. For instance,
tile 4 has edge 1 on the North, 2 on the East, 0 on the South
and West.

As the 2x2 tiles have to overlap in the previous state,
that is tantamount to matching these edges. So, for instance,
tile 1 can be to the left of tile 2, because tile 1 has East-1
and tile 2 has West-1.

The solution below was obtained with 50% planning and
50% trial and error, but when all's said and done, it's just
divide and conquer and backtracking with some tweaks to make it work.
"""

from functools import total_ordering

# A few constants used in the layered divide-and-conquer
MAX_ITERATIONS = 300
MAX_N = 5
MAX_AREA = 5

# little helper functions and cosntants that make life easier.
# self-explanatory

DIRECTIONS = list('NESW')


def opposite(d):
    # I know this looks stupid, but it's efficient.
    if d == 'N':
        return 'S'

    if d == 'S':
        return 'N'

    if d == 'E':
        return 'W'

    if d == 'W':
        return 'E'


def bit(v, n): return "{:04b}".format(v)[n]


class Tile:
    """Represents a single 2x2 tile in the 'previous' state.
    A Tile is initialized by its representative number, e.g.
    1 1
    0 1
    is initialized as Tile(0b1101), or Tile(13).
    """

    def __init__(self, v):
        self.v = v
        self.edges = {
            d: edge
            for d, edge in zip(list("NESW"), self._get_edge(v))
            }

    def __eq__(self, other):
        return self.v == other.v

    def __ne__(self, other):
        return self.v != other.v

    def __hash__(self):
        return hash(self.v)  # a tile is only unique wrt its number.

    def _get_edge(self, v):
        return (
            int(bit(v, 0) + bit(v, 1), 2),
            int(bit(v, 1) + bit(v, 3), 2),
            int(bit(v, 2) + bit(v, 3), 2),
            int(bit(v, 0) + bit(v, 2), 2),
        )


@total_ordering
class TileHub:
    """A TileHub represents all of the tiles that could
    occupy a given square at a given tile.
    As the game is solved, the TileHub loses tiles
    as they are filtered out because of adjacency restrictions.

    To allow for that filtering, the TileHub keeps track
    of its neighbors and is able to call upon them
    to update its own contents.
    """

    def __init__(self, v, true_tile, tileset):
        """Initializes the TileHub

        Args:
            v (int): A unique identifier within a Board
            true_tile (bool): Whether it originated from
                a True value in the original matrix
            tileset (set): All of the possible tiles
                at the moment of instantiation
        """
        self.v = v
        self.true_tile = true_tile
        self.tileset = tileset
        self.neighbors = dict()

    def __len__(self):
        return len(self.tileset)

    def __hash__(self):
        return hash(self.v)

    def __eq__(self, other):
        return self.v == other.v

    def __lt__(self, other):
        return self.v < other.v

    def copy(self):
        # used during backtracking to generate new instances
        return TileHub(self.v, self.true_tile, set(self.tileset))

    def set_neighbor(self, d, tilehub):
        self.neighbors[d] = tilehub

    def report_edge(self, d):
        # informs all available edges
        # in a given direction from its current tiles
        return {t.edges[d] for t in self.tileset}

    def filter_tiles(self, d, V):
        """Given a set V of edge numbers
        and a direction d, excludes all tiles
        in self.tileset whose edge d
        is not an element of V.
        Reports whether this resulted in a change
        to tileset.

        Args:
            d (str): a direction from N, E, S, W
            V (iterable): a set of edge numbers

        Returns:
            bool: True if there was a change to tileset
        """
        prev_len = len(self.tileset)
        self.tileset = set(filter(lambda t: t.edges[d] in V, self.tileset))
        return len(self.tileset) != prev_len

    def update(self):
        """Calls upon its neighbors to report their edges
        and filters its tiles accordingly

        Returns:
            bool: Whether this resulted in change
        """
        if len(self) == 0:
            return False
        has_updated = False  # will become true if there is any change
        for d, n in self.neighbors.items():
            V = n.report_edge(opposite(d))
            has_updated = self.filter_tiles(d, V) or has_updated
        return has_updated


# all of the 16 possible tiles in the nebula problem
all_tiles = {
    True: [Tile(i) for i in [1, 2, 4, 8]],
    False: [Tile(i) for i in range(16) if i not in [1, 2, 4, 8]]
}


class Board:
    """The Board is a matrix of TileHubs.
    It is pushed and popped from the stack
    during the backtracking process.
    The Board can update itself when TileHubs are changed
    and create copies of itself for backtracking purposes.
    """

    def __init__(self, g, initialize=True):
        """Initializes a Board.

        Args:
            g (Matrix or two-dimension iterable): A Matrix object generated
            from the matrix of booleans in the nebula problem. If an iterable
            is supplied, then a Matrix is generated.
            initialize (bool, optional): Whether the board should
            update itself on instantiation. For optimization purposes.
            Defaults to True.
        """
        if isinstance(g, Matrix):
            g = g.g
        self.h = len(g)
        self.w = len(g[0])
        if isinstance(g[0][0], bool):
            self.board = [
                [
                    TileHub(i * self.w + j, b, all_tiles[b])
                    for j, b in enumerate(row)
                    ]
                for i, row in enumerate(g)
                ]
        else:
            self.board = g

        # Reverse dictionary where TileHubs are keys
        # and coordinates are values.
        self.coord_lookup = {
            self.board[i][j]: (i, j)
            for i in range(self.h)
            for j in range(self.w)
            }

        # Sets up adjacencies.
        self._build_neighborhood()

        # Keeps count of ones and zeros for solution-checking purposes.
        self.ones = len([t for row in self.board for t in row if len(t) == 1])
        self.zeros = len([t for row in self.board for t in row if len(t) == 0])

        if initialize:
            self.update_board()

    def __len__(self):
        return self.h * self.w

    def _move(self, i, j, d):
        """Answers the question, 'If I start at (i, j)
        and move one square in direction d,
        where will I land?

        Args:
            i (int): the starting y coordinate
            j (int): the starting x coordinate
            d (str): a direction from DIRECTIONS

        Returns:
            tuple: the resulting coordinates or None if illegal move
        """
        if d == 'N' and i > 0:
            return (i - 1, j)

        if d == 'W' and j > 0:
            return (i, j - 1)

        if d == 'S' and i < self.h - 1:
            return (i + 1, j)

        if d == 'E' and j < self.w - 1:
            return (i, j + 1)

    def copy(self):
        """Instances an exact copy of this board,
        but with a different id.

        Returns:
            Board: the copy
        """
        return Board([[t.copy() for t in row] for row in self.board], False)

    def near_solution(self):
        """We define a near-solution as a state wherein
        all but one TileHub has a tileset with a length of one,
        and exactly one TileHub has N > 1 tilesets.
        This indicates that the current Board represents
        exactly N solutions.

        Returns:
            bool: Whether or not this Board is at near-solution
        """
        return self.ones == self.h * self.w - 1 and self.zeros == 0

    def get_near_solution(self):
        """If at near-solution,
        returns the TileHub containing the multiple
        solving Tiles and its coordinates.

        Returns:
            (TileHub, (y_coordinate, x_coordinate)): The near-solution
            TileHub and its coordinates.
        """
        if self.near_solution():
            for key, value in self.coord_lookup.items():
                if len(key) > 1:
                    return key, value

    def update_count(self, prevlen, nowlen):
        """Updates the ones- and zeros-counters
        according to the previous and current length
        of an updated TileHub.

        Args:
            prevlen (int): The previous length before the update
            nowlen (int): The current length, after the update
        """

        if prevlen != nowlen:
            if prevlen == 1:
                self.ones -= 1
            if nowlen == 1:
                self.ones += 1
            if nowlen == 0:
                self.zeros += 1

    def _get_border(self, d):
        """Here the Board acknowledges that the Universe
        does not revolve around it and that it might just be
        a big Tile in a bigger Board. Hence it solicitously
        reports its overall border in a given direction
        by concatenating the corresponding edges of the tiles along that
        border. Meant only to be used when a Board is completely solved
        (i.e. only 1 tile per TileHub).

        Args:
            d (str): a direction from DIRECTIONS

        Returns:
            str: the concatenated edges, which is an edge per se
        """

        if d == 'N':
            return "".join(
                str(
                    list(self.board[0][j].tileset)[0].edges['N'])
                    for j in range(self.w)
                )

        if d == 'E':
            return "".join(
                str(
                    list(self.board[i][-1].tileset)[0].edges['E'])
                    for i in range(self.h)
                )

        if d == 'S':
            return "".join(
                str(
                    list(self.board[-1][j].tileset)[0].edges['S'])
                    for j in range(self.w)
                )

        if d == 'W':
            return "".join(
                str(
                    list(self.board[i][0].tileset)[0].edges['W'])
                    for i in range(self.h)
                )

    def get_borders(self):
        """Returns a dict of its consolidated borders
        so that the Board may be used as a Tile.

        Returns:
            dict: the concatenated border in each direction
        """
        return {d: self._get_border(d) for d in DIRECTIONS}

    def replace_tilehub(self, u, v, tilehub):
        """Replaces a TileHub at coordinates (u, v)
        with the tilehub supplied and makes the necessary adjustments.

        Args:
            u (int): the y-coordinate
            v (int): the x-coordinate
            tilehub (TileHub): the new tilehub

        Returns:
            TileHub: the new TileHub as replaced
        """

        # update zeros and ones
        prevlen = len(self.board[u][v])
        nowlen = len(tilehub)
        self.update_count(prevlen, nowlen)

        # update lookup dictionaries
        self.coord_lookup.pop(self.board[u][v], None)
        self.board[u][v] = tilehub
        self.coord_lookup[tilehub] = (u, v)

        # update neighbors
        for d in DIRECTIONS:
            coords = self._move(u, v, d)
            if coords:
                n = self.board[coords[0]][coords[1]]
                tilehub.set_neighbor(d, n)
                n.set_neighbor(opposite(d), tilehub)

        return tilehub

    def make_path(self):
        """Creates a predetermined sequence of TileHubs
        to be iterated over during backtracking. For reasons
        unknown, the best results obtained were by first going over
        the TileHubs with False tiles, then their neighbors, and then
        the rest.

        Returns:
            list: A list of all the TileHubs in the described sequence.
        """

        unvisited = list(t for row in self.board for t in row if t.true_tile)
        path = [
            self.coord_lookup[tilehub]
            for row in self.board
            for tilehub in row
            if not tilehub.true_tile
            ]

        for i, j in list(path):
            tilehub = self.board[i][j]
            for neighbor in tilehub.neighbors.values():
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    path.append(self.coord_lookup[neighbor])

        path += [self.coord_lookup[tilehub] for tilehub in unvisited]
        return path

    def _build_neighborhood(self):
        """Iterates over all TileHubs and assigns neighbors.
        """
        for i in range(self.h):
            for j in range(self.w):
                for d in DIRECTIONS:
                    coords = self._move(i, j, d)
                    if coords:
                        self.board[i][j].set_neighbor(
                            d, self.board[coords[0]][coords[1]]
                        )

    def update_board(self):
        """Iterates over all TileHubs and has them
        self-update according to their neighbors' edges.
        Repeats until no more changes are registered.
        """

        confirmed = False
        while not confirmed:
            updated = True

            while updated:
                updated = False
                for row in self.board:
                    for tilehub in row:
                        prevlen = len(tilehub)
                        this_updated = tilehub.update()
                        updated = this_updated or updated
                        nowlen = len(tilehub)
                        self.update_count(prevlen, nowlen)

            confirmed = True
            for row in self.board:
                for tilehub in row:
                    prevlen = len(tilehub)
                    if tilehub.update():
                        confirmed = False
                    nowlen = len(tilehub)
                    self.update_count(prevlen, nowlen)

    def update_local(self, tilehub):
        """Same as update(), but here we restrict the
        process to the TileHubs that are affected
        by a recently-replaced TileHub specified in the arguments.

        Args:
            tilehub (TileHub): Ideally a recently replaced TileHub.
        """

        root = tilehub
        visited = [root]
        queue = [root]
        backlog = []

        while queue:
            tilehub = queue.pop()
            if tilehub != root and len(tilehub) == 1:
                continue

            prevlen = len(tilehub)
            did_update = tilehub.update()
            nowlen = len(tilehub)
            self.update_count(prevlen, nowlen)

            # Board has become an impossible solution
            # and no further action is needed.
            if nowlen == 0:
                return

            if did_update:
                backlog.append(tilehub)

            if did_update or tilehub == root:
                for neighbor in tilehub.neighbors.values():
                    if neighbor not in visited:
                        visited.append(neighbor)
                        queue.insert(0, neighbor)

        for t in backlog:
            self.update_local(t)

    def is_solution(self):
        """Verifies whether this Board
        is a single solution.

        Returns:
            bool: Whether it is.
        """
        return self.zeros == 0 and self.ones == self.w * self.h

    def is_fail(self):
        """Verifies whether the Board
        cannot be solved.

        Returns:
            bool: Whether it cannot.
        """
        return self.zeros > 0


class Solution:
    """When a Board and its TileHubs love each other
    very much, they generate a Solution. A Solution is
    just a set of consolidated borders
    (see Board.getborders()) that result from a particular
    arrangement of tiles. Sometimes there is more than one
    solution that results in the same set of borders. A Solution
    object represents only one 'copy' of such a solution.
    """
    def __init__(self, edges):
        self.edges = edges
        # I know, I know, I should've subclassed dict.

    def __eq__(self, other):
        return all(self.edges[d] == other.edges[d] for d in DIRECTIONS)

    def __hash__(self):
        return hash("-".join(self.edges.values()))

    def merge(self, other, axis):
        """If a Solution can just be a big tile
        that's part of a larger board, then we can
        merge two solutions to create a solution
        for that bigger board.

        Args:
            other (Solution): the adjacent solution
            axis (str): 'h' if we're merging with a Solution on the RIGHT,
            'v' if we're merging with a Solution BELOW.

        Returns:
            Solution: the resulting merged solution
        """
        if axis == 'h':
            return Solution(
                {
                    'N': self.edges['N'] + other.edges['N'],
                    'E': other.edges['E'],
                    'W': self.edges['W'],
                    'S': self.edges['S'] + other.edges['S']
                    })

        elif axis == 'v':
            return Solution(
                {
                    'N': self.edges['N'],
                    'E': self.edges['E'] + other.edges['E'],
                    'W': self.edges['W'] + other.edges['W'],
                    'S': other.edges['S']
                    })


class GeneralSolution:
    """
    *salute* General Solution!

    A GeneralSolution is meant to be the set
    of all solutions that uniquely solve a given
    Board. We can then merge it to an adjacent
    GeneralSolution to obtain the GeneralSolution
    for the merged board. It's like sharing
    a club sandwich, but backwards.

    As the purpose of a GeneralSolution is to be merged
    with others, it stores its solutions in a dictionary
    indexed primarily by direction. That way, we can get
    all of the East borders of a GeneralSolution and compare them
    with the West borders of another, thereby merging them
    by matched borders.

    The uniqueborders dict deserves additional commentary.
    Its structure is

    uniqueborders
        direction: one of N, E, S, W
            border: a unique border in that direction
                solution: a given Solution with that border in that direction
                    value: the number of said solutions in this GeneralSolution

    Believe me: it makes merging easier.
    """

    def __init__(self):
        # for bookkeeping purposes, we always instantiate
        # GeneralSolutions empty.
        self.uniqueborders = {d: dict() for d in DIRECTIONS}
        self.solution_count = 0

    def add_solution(self, solution, how_many=1):
        """Adds a Solution and makes the necessary changes.
        It is also possible to specify how many of these solutions
        are being added.

        Args:
            solution (Solution): a solution.
            how_many (int, optional): The number of solutions. Defaults to 1.
        """

        for d in DIRECTIONS:
            border = solution.edges[d]
            if border not in self.uniqueborders[d]:
                self.uniqueborders[d][border] = dict()
                self.uniqueborders[d][border][solution] = how_many
            else:
                n = self.uniqueborders[d][border].get(solution, 0) + how_many
                self.uniqueborders[d][border][solution] = n
        self.solution_count += 1

    def export_consolidated_unique_borders(self, direction=None):
        """Used in the topmost layer of this program
        where we ditch the GeneralSolution objects
        and just keep the necessary information to avoid
        hogging RAM.

        If a direction is specified, generates a dictionary
        of the unique borders in that directions and the number of times
        they occur. Otherwise, generates all of the unique west-east pairs
        and their occurrences.

        It's structured that way because in the final step,
        we merge the boards laterally.

        Args:
            direction (str, optional): One of DIRECTIONS. Defaults to None.

        Returns:
            dict: A dict of borders or border pairs and their occurrences.
        """

        if direction is not None:
            return {key:
                    sum(self.uniqueborders[direction][key].values())
                    for key in self.uniqueborders[direction]}

        # otherwise export west-east pairs
        export = dict()
        for west_border, solution_group in self.uniqueborders['W'].items():
            for solution, solution_count in solution_group.items():
                east_border = solution.edges['E']
                key = (west_border, east_border)
                export[key] = export.get(key, 0) + solution_count

        return export

    def merge(self, other, axis):
        """Merges this GeneralSolution to another
        along the given axis. Returns the resulting
        GeneralSolution

        Args:
            other (GeneralSolution): another GeneralSolution
            axis (str): 'h' if merging with the GeneralSolution
            to the RIGHT, 'v' if merging with the GeneralSolution BELOW.

        Returns:
            GeneralSolution: the merged GeneralSolution
        """
        if axis == 'v':
            edge1, edge2 = 'S', 'N'
        else:
            edge1, edge2 = 'E', 'W'

        matching_borders = set(self.uniqueborders[edge1])\
            .intersection(other.uniqueborders[edge2])
        gs = GeneralSolution()
        for matching_border in matching_borders:
            for s1 in self.uniqueborders[edge1][matching_border]:
                for s2 in other.uniqueborders[edge2][matching_border]:
                    n1 = self.uniqueborders[edge1][matching_border][s1]
                    n2 = other.uniqueborders[edge2][matching_border][s2]
                    gs.add_solution(s1.merge(s2, axis), n1 * n2)
        return gs

    def count(self):
        """Returns the absolute number of solutions
        this GeneralSolution represents. When called upon,
        it is expected to represent the definitive solution
        to a board.

        Returns:
            int: The number of solutions.
        """
        return self.solution_count


class ZeroSolution(GeneralSolution):
    """Used to save space. A failed GeneralSolution.
    """

    def __init__(self):
        super(ZeroSolution, self).__init__()

    def merge(self, other, axis):
        return ZeroSolution()


class Matrix:
    """A wrapper class for the initializing Matrix
    given in the tests cases. Cuts itself into pieces
    and does all sorts of nasty stuff.
    """

    def __init__(self, initializer=None):
        if isinstance(initializer, list):
            self.g = initializer
        elif isinstance(initializer, str):
            self.g = self.string_to_matrix(initializer)
        self.h = len(self.g)
        self.w = len(self.g[0])

    def all_false(self):
        return not any(v for row in self.g for v in row)

    def matrix_to_string(self):
        return " ".join("".join(str(int(v)) for v in row) for row in self.g)

    def string_to_matrix(self, s):
        return [[bool(int(c)) for c in list(row)] for row in s.split(" ")]

    def find_biggest_false_rectangle(self):
        """The idea here is that big swathes of False values
        make the number of solutions intractable, and thus
        by splitting them, we can manage the solution more easily.
        Finds the biggest rectangle of False values and returns its
        coordinates and dimensions.

        Returns:
            dict: A dictionary of the biggest rectangle, containing
            its area, its coordinates, and its dimensions.
        """

        # adapted from https://stackoverflow.com/questions/2478447/find-largest-rectangle-containing-only-zeros-in-an-n%C3%97n-binary-matrix

        from itertools import groupby

        F = [[0 for _ in range(self.w)] for _ in range(self.h)]
        F[0] = [int(not v) for v in self.g[0]]

        for i in range(1, self.h):
            for j in range(self.w):
                F[i][j] = 1 + F[i - 1][j] if not self.g[i][j] else 0
        all_groups = []

        for i in range(self.h - 1, -1, -1):
            row = F[i]
            G = groupby(row, key=lambda v: v > 0)
            groups = []
            k = 0
            for _, group in G:
                group = list(group)
                groups.append((i, k, group))
                k += len(group)

            all_groups.append(groups)

        max_area = float("-inf")
        max_area_i = max_area_j = max_area_w = max_area_h = -1

        for row in all_groups:
            for u, v, group in row:

                if sum(group) == 0:
                    continue

                for i in range(len(group)):
                    for j in range(i + 1, len(group) + 1):
                        this_cut = group[i:j]
                        this_h = min(this_cut)
                        this_w = len(this_cut)
                        this_area = this_h * this_w
                        if this_area > max_area:
                            max_area_w = this_w
                            max_area_h = this_h
                            max_area_i = u - this_h + 1
                            max_area_j = v + i
                            max_area = this_area

        return {
            'max_area': max_area,
            'coords': (max_area_i, max_area_j),
            'dimensions': (max_area_h, max_area_w)
            }

    def cleave_matrix(self):
        """Divides the matrix into two submatrices,
        usually in order to divide the biggest area
        of False values.

        Returns:
            (Matrix, Matrix, str): The two resulting Matrix objects
            and the axis of the division ('h' if left-right, 'v' if top-down)
        """

        if self.h == 1:
            cut = self.w // 2
            axis = 'h'
        elif self.w == 1:
            cut = self.h // 2
            axis = 'v'
        else:
            maxrec = self.find_biggest_false_rectangle()
            h, w = maxrec['dimensions']

            if h == 1:
                axis = 'h'
            elif w == 1:
                axis = 'v'
            else:
                axis = 'v' if h > w else 'h'
            u, v = maxrec['coords']

            if axis == 'v':
                cut = max(u + h // 2, 1)
                cut = min(cut, self.h - 1)

            else:
                cut = max(v + w // 2, 1)
                cut = min(cut, self.w - 1)

        if axis == 'v':
            return Matrix(self.g[:cut]), Matrix(self.g[cut:]), axis
        else:
            return (
                Matrix([row[:cut] for row in self.g]),
                Matrix([row[cut:] for row in self.g]),
                axis
                )


# LEVEL 1: Solve a Board. This is the lowest level.
# We use generic backtracking.

def solve(starting_board):
    """Solves a Board or refuses to do so.

    Args:
        starting_board (Board): A Board to be solved.

    Returns:
        GeneralSolution: The GeneralSolution for the board or
        None if it takes too many iterations.
    """
    step = 0
    path = starting_board.make_path()
    partial_solution = (step, starting_board)
    stack = [partial_solution]
    gs = GeneralSolution()
    iterations = 0

    while stack:
        iterations += 1
        if iterations > MAX_ITERATIONS:
            # This is too hard and needs to be broken up more.
            return None
        step, b = stack.pop()

        u, v = path[step]  # always represents the TileHub being replaced.
        this_hub = b.board[u][v]

        # Case 1: The TileHub is already narrowed down to one.
        if len(this_hub) == 1:
            new_b = b.copy()

            if new_b.is_solution():
                gs.add_solution(Solution(new_b.get_borders()))
            elif not new_b.is_fail():
                stack.append((step + 1, new_b))

            continue

        # Case 2: The Board itself is 1x1.
        if b.w == 1 and b.h == 1:
            for tile in this_hub.tileset:
                new_b = b.copy()
                tilehub = TileHub(this_hub.v, this_hub.true_tile, {tile})
                new_b.replace_tilehub(u, v, tilehub)

                if new_b.is_solution():
                    gs.add_solution(Solution(new_b.get_borders()))
            continue

        # Case 3: The TileHub to be replaced has multiple tiles.
        for tile in this_hub.tileset:
            new_b = b.copy()
            tilehub = TileHub(this_hub.v, this_hub.true_tile, {tile})
            new_b.replace_tilehub(u, v, tilehub)

            # See how replacing the TileHub affects the board.
            # This allows us to weed out failed backtracking branches
            # way beforehand.
            new_b.update_local(tilehub)

            if new_b.near_solution():
                # if we get here, it means that the next backtracking
                # step in this branch would result in solutions
                # so we deal with them instead of branching forth
                next_hub, next_coords = new_b.get_near_solution()
                next_u, next_v = next_coords

                for next_tile in next_hub.tileset:
                    new_new_b = new_b.copy()
                    next_tilehub = TileHub(
                        next_hub.v,
                        next_hub.true_tile,
                        {next_tile}
                        )
                    new_new_b.replace_tilehub(next_u, next_v, next_tilehub)
                    new_new_b.update_local(next_tilehub)
                    if new_new_b.is_solution():
                        gs.add_solution(Solution(new_new_b.get_borders()))

            elif not new_b.is_fail():
                stack.append((step + 1, new_b))

    if gs.count() == 0:
        return ZeroSolution()
    else:
        return gs


# Level 2: Preprocessing. Here we take Boards and subdivide them
# if they take too long.

known_solutions = dict()


def preprocess(g):
    """Assesses whether a Matrix should be further subdivided
    or can be solved directly.

    Args:
        g (Matrix): a matrix to be solved.

    Returns:
        GeneralSolution: the general solution to that Matrix.
    """
    s = g.matrix_to_string()
    gs = None
    if g.h <= MAX_N and g.w <= MAX_N:
        if s in known_solutions:
            gs = known_solutions[s]
        else:
            gs = solve(Board(g))
            if gs is not None:
                known_solutions[s] = gs

    if gs is None:
        g1, g2, axis = g.cleave_matrix()

        gs1 = preprocess(g1)
        gs2 = preprocess(g2)

        if s in known_solutions:
            gs = known_solutions[s]
        else:
            gs = gs1.merge(gs2, axis)
            known_solutions[s] = gs

    return gs

# Level 2: The grid. Here we try to restrict subdivisions to at most 3x3 Boards
# so we can reuse common solutions (all-False 3x3 is particularly troublesome).


def bottom_up_divide(g):
    columns = [[row[i:i+3] for row in g] for i in range(0, len(g[0]), 3)]
    grid = [[column[i:i+3] for i in range(0, len(g), 3)] for column in columns]
    return grid


def grid_it(g):
    grid = bottom_up_divide(g)

    solution = preprocess(Matrix(grid[0][0]))
    for submatrix in grid[0][1:]:
        partial = preprocess(Matrix(submatrix))
        solution = solution.merge(partial, 'v')

    for col in grid[1:]:
        this_column_solution = preprocess(Matrix(col[0]))

        for submatrix in col[1:]:
            partial = preprocess(Matrix(submatrix))
            this_column_solution = this_column_solution.merge(partial, 'v')
        solution = solution.merge(this_column_solution, 'h')

    return solution


# Level 4: Horizontal merging. The topmost level where we start
# dooming GeneralSolutions to oblivion and crunching numbers.

def solution(g):

    MAX_WIDTH = 4  # works well. Ain't broke, not fixing.

    width = len(g[0])
    if width <= MAX_WIDTH:
        return preprocess(Matrix(g)).count()

    # From here on, we solve from west to east,
    # getting the total number of solutions.

    submatrices = [
        [row[i:i+MAX_WIDTH] for row in g]
        for i in range(0, width, MAX_WIDTH)
        ]

    east_borders = grid_it(submatrices[0])\
        .export_consolidated_unique_borders('E')

    for i in range(1, len(submatrices) - 1):
        this_node = grid_it(submatrices[i])\
            .export_consolidated_unique_borders()
        new_east_borders = dict()
        for key, value in this_node.items():
            west, east = key
            if west in east_borders:
                new_value = new_east_borders.get(east, 0)
                new_value += east_borders[west] * value
                new_east_borders[east] = new_value
        east_borders = new_east_borders

    west_borders = grid_it(submatrices[-1])\
        .export_consolidated_unique_borders('W')

    return sum(
        east_borders[matching_border] * west_borders[matching_border]
        for matching_border
        in set(east_borders.keys()).intersection(west_borders.keys())
    )
