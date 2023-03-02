from copy import deepcopy
from heapq import heappush, heappop
import time
import argparse
import sys
from typing import Optional, Set

# ==============================================================================

char_goal = '1'
char_single = '2'
Empty = '.'

UP = '^'
Down = 'v'
Left = '<'
Right = '>'


class Piece:
    """
    This represents a piece on the Hua Rong Dao puzzle.
    """
    is_goal: bool
    is_single: bool
    col: int
    row: int
    orientation: str

    def __init__(self, is_goal: bool, is_single: bool, col: int,
                 row: int, orientation: Optional[str] = None):
        """
        :param is_goal: True if the piece is the goal piece and False otherwise.
        :type is_goal: bool
        :param is_single: True if this piece is a 1x1 piece and False otherwise.
        :type is_single: bool
        :param col: The x coordinate of the top left corner of the piece.
        :type col: int
        :param row: The y coordinate of the top left corner of the piece.
        :type row: int
        :param orientation: The orientation of the piece (one of 'h' or 'v')
            if the piece is a 1x2 piece. Otherwise, this is None
        :type orientation: str
        """

        self.is_goal = is_goal
        self.is_single = is_single
        self.col = col
        self.row = row
        self.orientation = orientation

    def __repr__(self):
        return '{} {} {} {} {}'.format(self.is_goal, self.is_single,
                                       self.row, self.col,
                                       self.orientation)

    def copy(self) -> 'Piece':
        return Piece(self.is_goal, self.is_single, self.col, self.row,
                     self.orientation)

    def move(self, direction: str) -> None:
        """v ^ > < """
        if direction == UP:
            if self.row == 0:
                raise ValueError('row 0 try to move up')
            self.row -= 1
        elif direction == Down:
            if self.row == 4:
                raise ValueError('row 3 try to move down')
            self.row += 1
        elif direction == Left:
            if self.col == 0:
                raise ValueError('col 0 try to move left')
            self.col -= 1
        elif direction == Right:
            if self.col == 3:
                raise ValueError('col 0 try to move right')
            self.col += 1
        else:
            raise ValueError("Invalid direction!\n")


class Board:
    """
    Board class for setting up the playing board.

    ===Representation Invariant===
    grid:[
        [^11^],
        [v11v],
        [^<>^],
        [v22v],
        [2..2]
        ]
    """
    width: int
    height: int
    pieces: list[Piece]
    goal_piece: Piece
    grid: list[list[str]]
    empty_coords: list[list[int]]

    def __init__(self, pieces: list[Piece]):
        """
        :param pieces: The list of Pieces
        :type pieces: List[Piece]
        """

        self.width = 4
        self.height = 5

        self.pieces = pieces

        # self.grid is a 2-d (size * size) array automatically generated
        # using the information on the pieces when a board is being created.
        # A grid contains the symbol for representing the pieces on the board.
        self.grid = []
        self.empty_coords = []
        self.__construct_grid()

    def __construct_grid(self):
        """
        Called in __init__ to set up a 2-d grid based on the piece location
        information.

        """
        self.grid = []
        for i in range(self.height):
            line = []
            for j in range(self.width):
                line.append('.')
            self.grid.append(line)

        for piece in self.pieces:
            if piece.is_goal:
                self.goal_piece = piece
                self.grid[piece.row][piece.col] = char_goal
                self.grid[piece.row][piece.col + 1] = char_goal
                self.grid[piece.row + 1][piece.col] = char_goal
                self.grid[piece.row + 1][piece.col + 1] = char_goal
            elif piece.is_single:
                self.grid[piece.row][piece.col] = char_single
            else:
                if piece.orientation == 'h':
                    self.grid[piece.row][piece.col] = '<'
                    self.grid[piece.row][piece.col + 1] = '>'
                elif piece.orientation == 'v':
                    self.grid[piece.row][piece.col] = '^'
                    self.grid[piece.row + 1][piece.col] = 'v'

        self.empty_coords = []
        for row in range(self.height):
            for col in range(self.width):
                if self.grid[row][col] == Empty:
                    self.empty_coords.append([row, col])

    def __str__(self) -> str:
        """
        Print out the current board.

        """
        s = ''
        grid = self.grid
        for row in grid:
            for ch in row:
                s += ch
            s += '\n'
        return s

    def copy(self) -> 'Board':
        pieces_cpy = []
        for piece in self.pieces:
            pieces_cpy.append(piece.copy())
            new_board = Board(pieces_cpy)
        return Board(pieces_cpy)

    def _check_h_consec_empty(self) -> bool:
        """return true iff two empty space is horizontally consecutively
        aligned."""
        empty_coor1 = self.empty_coords[0]
        empty_coor2 = self.empty_coords[1]
        return empty_coor1[0] == empty_coor2[0] \
            and abs(empty_coor2[1] - empty_coor1[1]) == 1

    def _check_v_consec_empty(self) -> bool:
        """return true iff two empty space is vertically consecutively
        aligned."""
        empty_coor1 = self.empty_coords[0]
        empty_coor2 = self.empty_coords[1]
        return empty_coor1[1] == empty_coor2[1] \
            and abs(empty_coor2[0] - empty_coor1[0]) == 1

    def _check_mobility(self, piece: Piece, is_verbose: bool = False) \
            -> list[str]:
        """ Return directions iff piece is movable, otherwise False.

        """

        empty_coor1 = self.empty_coords[0]
        empty_coor2 = self.empty_coords[1]
        rslt = []

        if piece.is_single:
            # check empty around
            piece_left = [piece.row, piece.col - 1]
            piece_right = [piece.row, piece.col + 1]
            piece_abv = [piece.row - 1, piece.col]
            piece_blw = [piece.row + 1, piece.col]
            if piece_left == empty_coor2 or piece_left == empty_coor1:
                rslt.append(Left)
            if piece_right == empty_coor2 or piece_right == empty_coor1:
                rslt.append(Right)
            if piece_abv == empty_coor2 or piece_abv == empty_coor1:
                rslt.append(UP)
            if piece_blw == empty_coor2 or piece_blw == empty_coor1:
                rslt.append(Down)
            return rslt

        are_emptys_v = self._check_v_consec_empty()
        are_emptys_h = self._check_h_consec_empty()

        top_empty = []
        if are_emptys_v:
            top_empty = empty_coor1 if empty_coor1[0] < empty_coor2[0] \
                else empty_coor2

        left_empty = []
        if are_emptys_h:
            left_empty = empty_coor1 if empty_coor1[1] < empty_coor2[1] \
                else empty_coor2

        if piece.is_goal:
            if are_emptys_v:
                if is_verbose:
                    print(f'vertical consecutive alignment detected')
                piece_left = [piece.row, piece.col - 1]
                piece_right = [piece.row, piece.col + 2]
                if piece_left == top_empty:
                    rslt.append(Left)
                elif piece_right == top_empty:
                    rslt.append(Right)
                return rslt

            if are_emptys_h:
                if is_verbose:
                    print(f'horizontal consecutive alignment detected')
                piece_abv = [piece.row - 1, piece.col]
                piece_blw = [piece.row + 2, piece.col]
                if piece_abv == left_empty:
                    rslt.append(UP)
                if piece_blw == left_empty:
                    rslt.append(Down)
                return rslt
            return []

        # 2x1 piece
        if piece.orientation == 'v':
            if are_emptys_v:
                # can't move up or down
                piece_left = [piece.row, piece.col - 1]
                piece_right = [piece.row, piece.col + 1]
                if piece_left == top_empty:
                    rslt.append(Left)
                elif piece_right == top_empty:
                    rslt.append(Right)
                return rslt
            # check movable up and down
            piece_abv = [piece.row - 1, piece.col]
            piece_blw = [piece.row + 2, piece.col]
            if piece_abv == empty_coor2 or piece_abv == empty_coor1:
                rslt.append(UP)
            if piece_blw == empty_coor2 or piece_blw == empty_coor1:
                rslt.append(Down)
            return rslt

        if piece.orientation == 'h':
            if are_emptys_h:
                # can't move left or right
                piece_abv = [piece.row - 1, piece.col]
                piece_blw = [piece.row + 1, piece.col]
                if piece_abv == left_empty:
                    rslt.append(UP)
                elif piece_blw == left_empty:
                    rslt.append(Down)
                return rslt
            # check left and right
            piece_left = [piece.row, piece.col - 1]
            piece_right = [piece.row, piece.col + 2]
            if piece_left == empty_coor2 or piece_left == empty_coor1:
                rslt.append(Left)
            if piece_right == empty_coor2 or piece_right == empty_coor1:
                rslt.append(Right)
            return rslt

    def is_solved(self) -> bool:
        return self.goal_piece.row == 3 and self.goal_piece.col == 1

    def next_moves(self) -> list['Board']:
        """Return a list of all next potential moves unless its goal.
        """
        if self.is_solved():
            return []
        rslt = []
        for i in range(len(self.pieces)):
            valid_moves = self._check_mobility(self.pieces[i])
            for direction in valid_moves:
                new_board = self.copy()
                new_board.pieces[i].move(direction)
                new_board.__construct_grid()
                rslt.append(new_board)
        return rslt


class State:
    """
    State class wrapping a Board with some extra current state information.
    Note that State and Board are different. Board has the locations of the
    pieces. State has a Board and some extra information that is relevant to
    the search: heuristic function, f value, current depth and parent.
    """
    board: Board
    f: int
    depth: int
    parent: Optional['State']
    id: int

    def __init__(self, _board: Board, depth: int,
                 parent: Optional['State'] = None):
        """
        :param _board: The board of the state.
        :param f: The f value of current state
        :param depth: The depth of current state in the search tree.
        :param parent: The parent of current state.
        :type parent: Optional[State]
        """
        self.board = _board
        self.depth = depth
        self.parent = parent
        self.id = hash(_board)  # The id for breaking ties.

    def __repr__(self) -> str:
        return f'State id: {self.id % 100}'

    def __lt__(self, other):
        if isinstance(other, State):
            if self.f < other.f:
                return True
            if self.f == other.f:
                return self.id < other.id
            return False
        raise ValueError('not same type!')

    def next(self) -> list['State']:
        """next possible states
        """
        if self.board.is_solved():
            return []
        next_boards = self.board.next_moves()
        rslt = []
        for board in next_boards:
            new_state = State(board, self.depth + 1, self)
            rslt.append(new_state)
        return rslt

    def is_goal(self) -> bool:
        return self.board.is_solved()

    def _heuristic(self) -> int:
        curr_coor = (self.board.goal_piece.row, self.board.goal_piece.col)
        # goal coor = (3,1)
        return abs(curr_coor[0] - 3) + abs(curr_coor[1] - 1)

    def compute_f(self) -> None:
        self.f = self._heuristic()  # + len(dfs(self)) - 1


def dfs(state: State) -> list[State]:
    frontier = [state]
    seen = set()
    trace = []
    while len(frontier) > 0:
        curr = frontier.pop(0)
        if curr.is_goal():
            trace.append(curr)
            break
        if str(curr.board) in seen:
            continue
        seen.add(str(curr.board))
        trace.append(curr)
        for next_state in curr.next():
            frontier.append(next_state)

    # get rslt from trace
    if not trace[-1].is_goal():
        print(f'trace length: {len(trace)}.\n'
              f'frontier empty? {len(frontier) == 0}')
        print('No solution found !')
        print(trace[-1].board.__str__())
        return []

    goal = trace[-1]
    rslt = [goal]
    parent = goal.parent
    while parent is not None:
        rslt.insert(0, parent)
        parent = parent.parent
    return rslt


def a_star(_state: State) -> list[State]:
    _state.compute_f()
    frontier = []
    heappush(frontier, _state)
    # for _next in state.next():
    #     heappush(frontier, _next)
    seen = set()
    trace = []
    while len(frontier) > 0:
        curr = heappop(frontier)
        if str(curr.board) in seen:
            continue
        seen.add(str(curr.board))
        if curr.is_goal():
            trace.append(curr)
            break
        trace.append(curr)
        for next_state in curr.next():
            next_state.compute_f()
            heappush(frontier, next_state)

    # get rslt from trace
    if not trace[-1].is_goal():
        print(f'trace length: {len(trace)}')
        raise Exception('No solution found !')

    goal = trace[-1]
    rslt = [goal]
    parent = goal.parent
    while parent is not None:
        rslt.insert(0, parent)
        parent = parent.parent
    return rslt


def output_to_file(rslt: list[State], filename: str) -> None:
    file = open(filename, 'w')
    for state in rslt:
        grid = state.board.grid
        for row in grid:
            file.writelines(row + ['\n'])
        file.write('\n')
    file.close()


def read_from_file(filename) -> Board:
    """
    Load initial board from a given file.

    :param filename: The name of the given file.
    :type filename: str
    :return: A loaded board
    :rtype: Board
    """

    puzzle_file = open(filename, "r")

    line_index = 0
    pieces = []
    g_found = False

    for line in puzzle_file:

        for x, ch in enumerate(line):

            if ch == '^':  # found vertical piece
                pieces.append(Piece(False, False, x, line_index, 'v'))
            elif ch == '<':  # found horizontal piece
                pieces.append(Piece(False, False, x, line_index, 'h'))
            elif ch == char_single:
                pieces.append(Piece(False, True, x, line_index, None))
            elif ch == char_goal:
                if not g_found:
                    pieces.append(Piece(True, False, x, line_index, None))
                    g_found = True
        line_index += 1

    puzzle_file.close()

    board = Board(pieces)

    return board


if __name__ == "__main__":
    board = read_from_file("mid.txt")
    print(board)
    # boards = board.next_moves()
    init_state = State(board, 0)
    rslt = dfs(init_state)
    print(len(rslt))
    # output_to_file(rslt, 'a.txt')

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--inputfile",
    #     type=str,
    #     required=True,
    #     help="The input file that contains the puzzle."
    # )
    # parser.add_argument(
    #     "--outputfile",
    #     type=str,
    #     required=True,
    #     help="The output file that contains the solution."
    # )
    # parser.add_argument(
    #     "--algo",
    #     type=str,
    #     required=True,
    #     choices=['astar', 'dfs'],
    #     help="The searching algorithm."
    # )
    # args = parser.parse_args()
    #
    # # read the board from the file
    # board = read_from_file(args.inputfile)
    # init_state = State(board, 0)
    # rslt = []
    # if args.algo == 'dfs':
    #     rslt = dfs(init_state)
    # if args.algo == 'astar':
    #     rslt = a_star(init_state)
    # output_to_file(rslt, args.outputfile)
