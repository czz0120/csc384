import argparse
import copy
import sys
import time

cache = {}  # you can use this to implement state caching!

RED = 'rR'
RKING = 'R'
BLK = 'bB'
BKING = 'B'
Empty = '.'


class Piece:
    """
    This represents a piece on the Checker game.
    """
    is_red: bool
    is_black: bool
    col: int
    row: int
    is_king: bool

    def __init__(self, is_red: bool, is_black: bool, col: int,
                 row: int):
        """
        """
        self.is_red = is_red
        self.is_black = is_black
        self.col = col
        self.row = row
        self.is_king = False

    def __repr__(self):
        color = 'red' if self.is_red else 'black'
        king = "king" if self.is_king else "Not king"
        return '{} {} {} {}'.format(color, king,
                                    self.row, self.col)

    # def copy(self) -> 'Piece':
    # return Piece(self.is_goal, self.is_single, self.col, self.row,
    # self.orientation)


class Red_Piece(Piece):
    def __init__(self, row: int, col: int) -> None:
        super().__init__(True, False, col, row)


class Blk_Piece(Piece):
    def __init__(self, row: int, col: int) -> None:
        super().__init__(False, True, col, row)


class State:
    # This class is used to represent a state.
    # board : a list of lists that represents the 8*8 board
    """
    ===Representation Invariant===
    grid:[
        [.*.*.*.*], row 0: 1,3,5,7
        [*.*.*.*.], row 1: 0,2,4,6
        [.*.*.*.*],
        [*.*.*.*.],
        [.*.*.*.*],
        [*.*.*.*.],
        [.*.*.*.*],
        [*.*.*.*.]
        ]
    """
    width: int
    height: int
    board: list[list[str]]
    red_pieces: list[Piece]
    blk_pieces: list[Piece]

    def __init__(self, board):
        self.board = board
        self.width = 8
        self.height = 8
        for row in range(self.height):
            for col in range(self.width):
                char = self.board[row][col]
                if char in 'rR':
                    piece = Red_Piece(row, col)
                    if char.isupper():
                        piece.is_king = True
                    self.red_pieces.append(piece)
                elif char in 'bB':
                    piece = Blk_Piece(row, col)
                    if char.isupper():
                        piece.is_king = True
                    self.blk_pieces.append(piece)

    def _can_jump_left_up(self, piece: Piece) -> bool:

        bridge = 'bB' if piece.is_red else 'rR'
        l_up = (piece.row - 1, piece.col - 1)
        if self.board[l_up[0]][l_up[1]] in bridge and \
                l_up[0] - 1 >= 0 and l_up[1] - 1 >= 0:
            if self.board[l_up[0] - 1][l_up[1] - 1] == '.':
                return True
        return False

    def _can_jump_right_up(self, piece: Piece) -> bool:
        bridge = 'bB' if piece.is_red else 'rR'
        r_up = (piece.row - 1, piece.col + 1)
        if self.board[r_up[0]][r_up[1]] in bridge and \
                r_up[0] - 1 >= 0 and r_up[1] + 1 < self.width:
            if self.board[r_up[0] - 1][r_up[1] + 1] == '.':
                return True
        return False

    def _can_jump_left_down(self, piece: Piece) -> bool:

        bridge = 'bB' if piece.is_red else 'rR'
        l_down = (piece.row + 1, piece.col - 1)
        if self.board[l_down[0]][l_down[1]] in bridge and \
                l_down[0] + 1 < self.height and l_down[1] - 1 >= 0:
            if self.board[l_down[0] + 1][l_down[1] - 1] == '.':
                return True
        return False

    def _can_jump_right_down(self, piece: Piece) -> bool:

        bridge = 'bB' if piece.is_red else 'rR'
        l_down = (piece.row + 1, piece.col + 1)
        if self.board[l_down[0]][l_down[1]] in bridge and \
                l_down[0] + 1 < self.height and l_down[1] + 1 >= self.width:
            if self.board[l_down[0] + 1][l_down[1] + 1] == '.':
                return True
        return False

    def _can_jump(self, piece: Piece) -> list[str]:
        rslt = []
        if piece.is_red or piece.is_king:
            if self._can_jump_left_up(piece):
                rslt.append('l_up')
                pass
            if self._can_jump_right_up(piece):
                rslt.append('r_up')
        if piece.is_black or piece.is_king:
            if self._can_jump_left_down(piece):
                rslt.append('l_down')
                pass
            if self._can_jump_right_down(piece):
                rslt.append('r_down')
        return rslt

    def jumps(self, piece: Piece) -> list[Piece]:
        rslt = []
        piece_cpy = copy.deepcopy(piece)
        if not self._can_jump(piece):
            return []

        for direction in self._can_jump(piece):
            new_piece = piece_cpy.jump(direction)
            rslt += self.jumps(new_piece)

    def _move_simple(self, piece: Piece) -> list[str]:
        pass

    def actions(self) -> list[str]:
        pass

    def display(self):
        for i in self.board:
            for j in i:
                print(j, end="")
            print("")
        print("")


def get_opp_char(player):
    if player in ['b', 'B']:
        return ['r', 'R']
    else:
        return ['b', 'B']


def get_next_turn(curr_turn):
    if curr_turn == 'r':
        return 'b'
    else:
        return 'r'


def read_from_file(filename):

    f = open(filename)
    lines = f.readlines()
    board = [[str(x) for x in l.rstrip()] for l in lines]
    f.close()

    return board


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzles."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    args = parser.parse_args()

    initial_board = read_from_file(args.inputfile)
    state = State(initial_board)
    turn = 'r'
    ctr = 0

    sys.stdout = open(args.outputfile, 'w')

    sys.stdout = sys.__stdout__
