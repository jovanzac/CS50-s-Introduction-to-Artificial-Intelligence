"""
Tic Tac Toe Player
"""
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_count, o_count = 0, 0
    for i in board :
        x_count += i.count(X)
        o_count += i.count(O)
    if x_count > o_count :
        return O
    else :
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    action_set = set()
    for i in range(3) :
        for j in range(3) :
            if board[i][j] == EMPTY :
                action_set.add((i,j))
    return action_set


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    user = player(board)
    new_board = copy.deepcopy(board)
    new_board[action[0]][action[1]] = user
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # looking for a winner across
    for i in board :
        if i.count(X) == 3 :
            return X
        elif i.count(O) == 3 :
            return O
    # Looking for a winner down
    for i in range(3) :
        if board[0][i] == X and board[1][i] == X and board[2][i] == X :
            return X
        elif board[0][i] == O and board[1][i] == O and board[2][i] == O :
            return O
    # Looking for a winner diagonally
    x_dia1, o_dia1, x_dia2, o_dia2 = 0, 0, 0, 0
    for i in range(3) :
        if board[i][i] == X :
            x_dia1 += 1
        elif board[i][i] == O :
            o_dia1 += 1
        if board[i][-(i+1)] == X :
            x_dia2 += 1
        elif board[i][-(i+1)] == O :
            o_dia2 += 1
    if x_dia1 == 3 or x_dia2 == 3 :
        return X
    elif o_dia1 == 3 or o_dia2 == 3 :
        return O
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    won = winner(board)
    if won == X or won == O :
        return True
    else :
        empty_count = 0
        for i in board :
            empty_count += i.count(EMPTY)
        if empty_count == 0 :
            return True
        else :
            return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    won = winner(board)
    if won == X :
        return 1
    elif won == O :
        return -1
    else :
        return 0


def minimising(board) :
    if terminal(board) :
        return utility(board), None
    v, move = 2, None
    for action in actions(board) :
        value,_ = maximising(result(board,action))
        if value < v :
            v, move = value, action
            if v == -1 :
                return v, move
    return v, move


def maximising(board) :
    if terminal(board) :
        return utility(board), None
    v, move = -2, None
    for action in actions(board) :
        value,_ = minimising(result(board,action))
        if value > v :
            v, move = value, action
            if v == 1 :
                return v, move
    return value, move


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board) :
        return None
    else :
        cur_player = player(board)
        if cur_player == X :
            max_val, action = maximising(board)
            return action
        elif cur_player == O :
            min_val, action = minimising(board)
            return action