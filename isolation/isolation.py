"""This file contains the `Board` class, which implements the rules for the
game Isolation as described in lecture, modified so that the players move like
knights in chess rather than queens.

It differs slightly from the implementation provided to students by using a
slightly more sophisticated timing system for each move, which allows it to
interrupt student agents who overrun the move time limit.
"""
import resource
import time

from copy import copy
from collections import namedtuple
from multiprocessing import Process, Queue, TimeoutError
from queue import Empty

__all__ = ['Board', 'play']

MatchResult = namedtuple("MatchResult", ["winner", "loser", "moves", "reason"])


def play(player1, player2, time_limit, move_timeout):
    """
    Execute a match between the players by alternately soliciting them to
    select a move and applying it in the game.

    Parameters
    ----------
    player1 : namedtuple("Player", "id, name, agent")
        player.agent is an instance of CustomPlayer, player.id is the
        database id

    player2 : namedtuple("Player", "id, name, agent")
        player.agent is an instance of CustomPlayer, player.id is the
        database id

    time_limit : numeric
        The maximum number of milliseconds to allow before timeout during
        each turn (see notes)

    move_timeout : numeric
        The number of seconds to wait before killing the search process
        (see notes)

    Returns
    -------
    (player, list<[(int, int),]>, str)
        Return multiple including the winning player, the complete game
        move history, and a string indicating the reason for losing (e.g.,
        timeout or invalid move).

    Notes
    -----
        (1) Each call to playerN.get_move is timed, and will result in
        a timeout if the run time exceeds time_limit milliseconds.  If the
        process running the search doesn't terminate within move_timeout
        _seconds_, then the process will be killed and the agent will
        forfeit.
    """
    game = Board(player1.agent, player2.agent)
    moves = []
    times = {player1.agent: [], player2.agent: []}

    while True:

        _player = game.active_player

        try:
            move, decision_time = _next_move(game, time_limit, move_timeout)
            moves.append(move)
            times[_player].append(decision_time)
        except TimeoutError:
            reason = "timeout"
            break
        except Exception as err:
            reason = str(err or "Unknown Exception")
            break

        if move not in game.get_legal_moves():
            if game.get_legal_moves():
                reason = "forfeit"
                break
            reason = "game over"
            break

        game.apply_move(move)

    if game.inactive_player == player1.agent:
        winner, loser = player1.id, player2.id
    elif game.inactive_player == player2.agent:
        winner, loser = player2.id, player1.id
    else:
        raise RuntimeError(
            "Invalid game result {} vs {}".format(player1.id, player2.id)
        )

    loser_times = times[game.active_player]
    winner_times = times[game.inactive_player]
    result = MatchResult(
        winner=(winner, sum(winner_times) / max(len(winner_times), 1)),
        loser=(loser, sum(loser_times) / max(len(loser_times), 1)),
        moves=list(zip(*[iter(moves)] * 2)),
        reason=reason
    )
    return result


def _next_move(game, time_limit, timeout):
    """Returns the utility of the current game state from the perspective of
    the specified player.

    Parameters:
    ----------
    game : Isolation.Board
        Current game state

    time_limit : numeric
        The maximum time (in milliseconds) to allow for each call to get_move;
        raises a timeout error if the elapsed time measured within the
        process exceeds this limit

    timeout : numeric
        The maximum time (in seconds) to wait for a response from get_move
        before killing the process and returning timeout

    Returns:
    ----------
    (int, int)
        A coordinate pair (row, column) indicating the next position for
        the active player on the board.
    """
    q = Queue()

    def run(player, state, *args, **kwargs):
        """
        Wrapper function to call player code in a separate process, which
        allows the code to be interrupted and killed on timeout.

        Parameters
        ----------
        player : object
            An object registered as a player in the current game with a
            get_move() method

        state : Isolation.Board
            Copy of the current game state with the inactive player object
            reference removed
        """
        def time_millis():
            return 1000 * time.process_time()

        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (0, 0))

        try:
            start_time = time_millis()
            move = player.get_move(
                state,
                time_left=lambda: time_limit - (time_millis() - start_time),
            )
            stop_time = time_millis()
            q.put((None, move, stop_time - start_time))
        except Exception as e:
            # pass exceptions out through the queue
            q.put((e, None, None))

    _game = game.copy()

    curr_move = None
    move_time = 0
    try:
        p = Process(target=run, args=(_game.active_player, _game))
        p.start()
        p.join(timeout=timeout)

        exception, curr_move, move_time = q.get_nowait()
        if exception:
            raise exception
        elif move_time > time_limit:
            raise TimeoutError("get_move timeout")
    except Empty:
        raise TimeoutError("Queue timeout")
    finally:
        if p and p.is_alive():
            p.terminate()

    return curr_move, move_time


class Board(object):
    """Implement a model for the game Isolation assuming each player moves like
    a knight in chess.

    Parameters
    ----------
    player_1 : object
        An object with a get_move() function. This is the only function
        directly called by the Board class for each player.

    player_2 : object
        An object with a get_move() function. This is the only function
        directly called by the Board class for each player.

    width : int (optional)
        The number of columns that the board should have.

    height : int (optional)
        The number of rows that the board should have.
    """
    BLANK = 0
    NOT_MOVED = None

    def __init__(self, player_1, player_2, width=7, height=7):
        self.width = width
        self.height = height
        self.move_count = 0
        self._player_1 = player_1
        self._player_2 = player_2
        self._active_player = player_1
        self._inactive_player = player_2

        # The last 3 entries of the board state includes initiative (0 for
        # player 1, 1 for player 2) player 2 last move, and player 1 last move
        self._board_state = [Board.BLANK] * (width * height + 3)
        self._board_state[-1] = Board.NOT_MOVED
        self._board_state[-2] = Board.NOT_MOVED

    def hash(self):
        return str(self._board_state).__hash__()

    @property
    def active_player(self):
        """The object registered as the player holding initiative in the
        current game state.
        """
        return self._active_player

    @property
    def inactive_player(self):
        """The object registered as the player in waiting for the current
        game state.
        """
        return self._inactive_player

    def get_opponent(self, player):
        """Return the opponent of the supplied player.

        Parameters
        ----------
        player : object
            An object registered as a player in the current game. Raises an
            error if the supplied object is not registered as a player in
            this game.

        Returns
        -------
        object
            The opponent of the input player object.
        """
        if player == self._active_player:
            return self._inactive_player
        elif player == self._inactive_player:
            return self._active_player
        raise RuntimeError("`player` must be an object registered as a player in the current game.")

    def copy(self):
        """ Return a deep copy of the current board. """
        new_board = Board(self._player_1, self._player_2, width=self.width, height=self.height)
        new_board.move_count = self.move_count
        new_board._active_player = self._active_player
        new_board._inactive_player = self._inactive_player
        new_board._board_state = copy(self._board_state)
        return new_board

    def forecast_move(self, move):
        """Return a deep copy of the current game with an input move applied to
        advance the game one ply.

        Parameters
        ----------
        move : (int, int)
            A coordinate pair (row, column) indicating the next position for
            the active player on the board.

        Returns
        -------
        isolation.Board
            A deep copy of the board with the input move applied.
        """
        new_board = self.copy()
        new_board.apply_move(move)
        return new_board

    def move_is_legal(self, move):
        """Test whether a move is legal in the current game state.

        Parameters
        ----------
        move : (int, int)
            A coordinate pair (row, column) indicating the next position for
            the active player on the board.

        Returns
        -------
        bool
            Returns True if the move is legal, False otherwise
        """
        idx = move[0] + move[1] * self.height
        return (0 <= move[0] < self.height and 0 <= move[1] < self.width and
                self._board_state[idx] == Board.BLANK)

    def get_blank_spaces(self):
        """Return a list of the locations that are still available on the board.
        """
        return [(i, j) for j in range(self.width) for i in range(self.height)
                if self._board_state[i + j * self.height] == Board.BLANK]

    def get_player_location(self, player):
        """Find the current location of the specified player on the board.

        Parameters
        ----------
        player : object
            An object registered as a player in the current game.

        Returns
        -------
        (int, int) or None
            The coordinate pair (row, column) of the input player, or None
            if the player has not moved.
        """
        if player == self._player_1:
            if self._board_state[-1] == Board.NOT_MOVED:
                return Board.NOT_MOVED
            idx = self._board_state[-1]
        elif player == self._player_2:
            if self._board_state[-2] == Board.NOT_MOVED:
                return Board.NOT_MOVED
            idx = self._board_state[-2]
        else:
            raise RuntimeError(
                "Invalid player in get_player_location: {}".format(player))
        w = idx // self.height
        h = idx % self.height
        return (h, w)

    def get_legal_moves(self, player=None):
        """Return the list of all legal moves for the specified player.

        Parameters
        ----------
        player : object (optional)
            An object registered as a player in the current game. If None,
            return the legal moves for the active player on the board.

        Returns
        -------
        list<(int, int)>
            The list of coordinate pairs (row, column) of all legal moves
            for the player constrained by the current game state.
        """
        if player is None:
            player = self.active_player
        return self.__get_moves(self.get_player_location(player))

    def apply_move(self, move):
        """Move the active player to a specified location.

        Parameters
        ----------
        move : (int, int)
            A coordinate pair (row, column) indicating the next position for
            the active player on the board.
        """
        idx = move[0] + move[1] * self.height
        last_move_idx = int(self.active_player == self._player_2) + 1
        self._board_state[-last_move_idx] = idx
        self._board_state[idx] = 1
        self._board_state[-3] ^= 1
        self._active_player, self._inactive_player = self._inactive_player, self._active_player
        self.move_count += 1

    def is_winner(self, player):
        """ Test whether the specified player has won the game. """
        return player == self._inactive_player and not self.get_legal_moves(self._active_player)

    def is_loser(self, player):
        """ Test whether the specified player has lost the game. """
        return player == self._active_player and not self.get_legal_moves(self._active_player)

    def utility(self, player):
        """Returns the utility of the current game state from the perspective
        of the specified player.

                    /  +infinity,   "player" wins
        utility =  |   -infinity,   "player" loses
                    \          0,    otherwise

        Parameters
        ----------
        player : object (optional)
            An object registered as a player in the current game. If None,
            return the utility for the active player on the board.

        Returns
        ----------
        float
            The utility value of the current game state for the specified
            player. The game has a utility of +inf if the player has won,
            a value of -inf if the player has lost, and a value of 0
            otherwise.
        """
        if not self.get_legal_moves(self._active_player):

            if player == self._inactive_player:
                return float("inf")

            if player == self._active_player:
                return float("-inf")

        return 0.

    def __get_moves(self, loc):
        """Generate the list of possible moves for an L-shaped motion (like a
        knight in chess).
        """
        if loc == Board.NOT_MOVED:
            return self.get_blank_spaces()

        r, c = loc
        directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                      (1, -2), (1, 2), (2, -1), (2, 1)]
        valid_moves = [(r + dr, c + dc) for dr, dc in directions
                       if self.move_is_legal((r + dr, c + dc))]
        return valid_moves

    def to_string(self, symbols=['1', '2']):
        """Generate a string representation of the current game state, marking
        the location of each player and indicating which cells have been
        blocked, and which remain open.
        """
        p1_loc = self._board_state[-1]
        p2_loc = self._board_state[-2]

        col_margin = len(str(self.height - 1)) + 1
        prefix = "{:<" + "{}".format(col_margin) + "}"
        offset = " " * (col_margin + 3)
        out = offset + '   '.join(map(str, range(self.width))) + '\n\r'
        for i in range(self.height):
            out += prefix.format(i) + ' | '
            for j in range(self.width):
                idx = i + j * self.height
                if not self._board_state[idx]:
                    out += ' '
                elif p1_loc == idx:
                    out += symbols[0]
                elif p2_loc == idx:
                    out += symbols[1]
                else:
                    out += '-'
                out += ' | '
            out += '\n\r'

        return out
