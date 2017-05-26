
import argparse
import enum
import glob
import json
import logging
import math
import os
import pickle
import random
import signal
import sqlite3
import time

import isolation

from collections import namedtuple
from contextlib import contextmanager
from datetime import datetime
from multiprocessing import TimeoutError
from multiprocessing.pool import ThreadPool


module_names = [
    'random', 'numpy', 'scipy', 'sklearn', 'itertools',
    'math', 'heapq', 'collections', 'array', 'copy', 'operator'
]
modules = {m: __import__(m) for m in module_names}

logfile = "{!s}.log".format(datetime.utcnow())
logging.basicConfig(filename=logfile,
                    format='%(asctime)s %(levelname)s::%(message)s',
                    datefmt='%y.%m.%d %H:%M:%S')
logger = logging.getLogger(__name__)

K = 32
MOVE_TIMEOUT = 10  # Time to wait (in seconds) before killing process
MOVE_TIME_LIMIT_MILLIS = 150  # Time limit before game is forfeit
IMP_TIME_LIMIT_SECONDS = 0.2  # Time limit before killing import
CONS_TIME_LIMIT_SECONDS = 10.  # Time limit before killing __init__
JSON_TIME_LIMIT_SECONDS = 300
ID_IMPROVED_PATH = "competition_agent.py"

ALPHANUM = [["{}{}".format(alpha, 7 - num)
             for alpha in "abcdefg"] for num in [6, 5, 4, 3, 2, 1, 0]][::-1]

Player = namedtuple("Player", "id, name, agent")
AgentRecord = namedtuple("AgentRecord", "id, pickle, rating, loadtime")
MatchRecord = namedtuple("MatchRecord", "id, round, agentA, agentB")
ResultRecord = namedtuple("ResultRecord", "id, winner_id, loser_id")


class Level(enum.Enum):
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0


def game2dict(player1, player2, positions):

    def toAlpha(pos):
        if not pos:
            return pos
        return ALPHANUM[int(pos[0])][int(pos[1])]

    def valid_position(pos):
        return isinstance(pos, tuple) and len(pos) == 2 and pos != (-1, -1)

    if isinstance(positions, str) or isinstance(positions, unicode):
        positions = eval(positions)
    _positions = list(filter(valid_position,
                      sum(list(map(list, positions)), [])))
    alpha_pos = list(map(toAlpha, _positions))
    moves = ["{}-{}".format(x, y)
             for x, y in zip(alpha_pos[:-2], alpha_pos[2:])]
    match = {
        "player1": str(player1),
        "player2": str(player2),
        "pos": dict(zip(alpha_pos[:2], ['wN', 'bN'])),
        "moves": moves,
        "locs": _positions
    }
    return match


@contextmanager
def timeout(seconds):
    """ """
    def handler(*args, **kwargs):
        logger.debug("TimeoutError in timeout context manager handler.")
        raise TimeoutError("Timeout after {} seconds".format(seconds))

    signal.signal(signal.SIGPROF, handler)
    signal.setitimer(signal.ITIMER_PROF, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)


def _validate_agent(agent_file):
    pathname, _ = os.path.splitext(agent_file)
    data_file = os.path.join(pathname, "data.json")

    data = None
    try:
        with timeout(JSON_TIME_LIMIT_SECONDS), open(data_file, 'rb') as df:
            data = json.load(df)
    except IOError as err:
        logger.debug(err)
    except Exception as err:
        logger.error(
            "Data failed on module {!s} with err: {}"
            .format(agent_file, err)
        )

    try:
        with timeout(IMP_TIME_LIMIT_SECONDS):
            module = __import__(pathname.replace('/', '.'),
                                globals=modules,
                                fromlist=("CustomPlayer"))
        logger.debug(
            "Successfully imported module {} in less than {} seconds"
            .format(module.__name__, IMP_TIME_LIMIT_SECONDS)
        )
    except Exception as err:
        logger.error(
            "Import failed on module {!s} with err: {}"
            .format(agent_file, err)
        )
        return None

    try:
        with timeout(CONS_TIME_LIMIT_SECONDS):
            start_time = time.process_time()
            new_agent = module.CustomPlayer(data=data, timeout=1.)
            end_time = time.process_time()
        logger.debug(
            "Successfully created {!s} agent in less than {} seconds"
            .format(new_agent, CONS_TIME_LIMIT_SECONDS)
        )
    except Exception as err:
        logger.error(
            "CustomPlayer creation failed with err {!s} for {}"
            .format(err, agent_file)
        )
        return None

    try:
        agent_path, _ = os.path.split(agent_file)
        pickle_path = os.path.join(agent_path, "agent.pickle")
        with open(pickle_path, 'wb') as f:
            pickle.dump(new_agent, f)

    except Exception as err:
        logger.error(
            "Pickle failed with err {!s} for {}"
            .format(err, agent_file)
        )
        return None

    try:
        board = isolation.Board(new_agent, "none")
        s_time = time.process_time() * 1000
        with timeout(1.5 * MOVE_TIME_LIMIT_MILLIS):
            first_move = new_agent.get_move(
                board,
                lambda: MOVE_TIME_LIMIT_MILLIS - (1000 * time.process_time() - s_time)
            )
        if first_move not in board.get_legal_moves():
            raise RuntimeError(
                "{} is not a valid move on an empty board".format(first_move))
        logger.debug(
            "{!s} agent returned a valid opening move."
            .format(new_agent)
        )
    except Exception as err:
        logger.error(
            "{} agent failed to return a valid first move with err {!s}"
            .format(agent_file, err)
        )
        return None

    return pickle_path, 1000 * (end_time - start_time) / CONS_TIME_LIMIT_SECONDS


def _add_agent(db_conn, pickle_path, time):
    cur = db_conn.cursor()
    cur.execute("INSERT INTO agents(pickle, loadtime) VALUES (?, ?)",
                (pickle_path, time))
    db_conn.commit()


def load_agents(db_conn, location):
    """Validate agents and add them to the database """
    logger.debug("Loading agents from {}".format(location))
    search_path = os.path.join(location, "*", "competition_agent.py")
    logger.debug("Searching for student files in {}".format(search_path))
    agent_files = glob.glob(search_path)
    logger.info("Adding {} agents to database".format(len(agent_files)))
    counter = 0
    for agent_file in agent_files:
        res = _validate_agent(agent_file)
        if res is None:
            continue
        _add_agent(db_conn, *res)
        counter += 1
        logger.debug("Successfully added agent from {!s}".format(agent_file))

    logger.info(
        "Finished adding {} agents out of {} files to database".format(
            counter, len(agent_files))
    )

    if counter % 2:
        res = _validate_agent(ID_IMPROVED_PATH)
        _add_agent(db_conn, *res)
        logger.info("Added an ID_Improved agent to make agent count even.")


def roundrobin(agents, round_num):
    """Round-robin tournament between at most four players """
    matches = []
    for idx, agent1 in enumerate(agents[:-1]):
        for agent2 in agents[idx + 1:]:
            logger.debug(
                "Adding Round Robin matches: {} vs {}".format(
                    agent1, agent2
                )
            )
            matches.append((round_num, agent1.id, agent2.id))
            matches.append((round_num, agent2.id, agent1.id))
    return matches


def random_pairings(agents, round_num):
    """Generate random pairings in an even-length list - allow repeats """
    N = len(agents) // 2
    random.shuffle(agents)
    _agents = agents[-N:]
    random.shuffle(_agents)
    matches = []
    for players in zip(agents[:N], agents[-N:], _agents):
        logger.debug(
            "Adding Random match between: {} vs {}".format(
                players[0], players[1]
            )
        )
        matches.append((round_num, players[0].id, players[1].id))

        logger.debug(
            "Adding Random match between: {} vs {}".format(
                players[2], players[0]
            )
        )
        matches.append((round_num, players[2].id, players[0].id))

    return matches


class AgentManager:
    """Database handler to manage tournament state

    Database tables include `agents`, `matches`, `results`, `history`, and
    `times`.

    The behavior of the manager depends on the tournament state. Tournament
    state is determined by the number of active agents:
        1 - tournament is over (there is a winner)
        4 - final four round robin
        N - ranked elimination

    If there are unfinished matches in the database, the manager will attempt
    to complete those before generating any new matches.  When all matches
    are completed, the manager will attempt to "finalize" the round by
    updating the score of every agent based on their performance in the round,
    and eliminate the lowest ranked agents until the number remaining is a
    power of two.
    """

    def __init__(self, db):
        self.db = db

    @property
    def winner(self):
        cur = self._active_agents()
        _winner = next(cur, None)
        try:
            next(cur)
            return None
        except StopIteration:
            pass
        return AgentRecord._make(_winner)

    @property
    def round(self):
        """Returns the current round number, or -1 if uninitialized """
        max_round = next(self.db.execute("SELECT MAX(round) FROM matches"))[0]
        return max_round if max_round is not None else -1

    def get_open_matches(self):
        """Returns a list of matches containing deserialized agents and matchID

        If there is already a winner, then return an empty list; if there are
        any matches in the database that have not yet been scored, return
        those; otherwise generate a new list of matches for the active
        players and return them
        """
        if self.winner:
            return []

        if not next(self._unfinished_matches(), (None,))[0]:  # check to resume
            self._finalize_round()
            self._make_matches()

        open_matches = list(map(MatchRecord._make, self._unfinished_matches()))
        matches = []
        for record in open_matches:
            match_id = record.id
            player1 = self._get_player_obj(record.agentA)
            player2 = self._get_player_obj(record.agentB)
            matches.append((player1, player2, match_id))
        return matches

    def record_match(self, match_id, match_result):
        """Store the results of a match in the DB """
        winner_id, w_time = match_result.winner
        loser_id, l_time = match_result.loser
        cur = self.db.cursor()
        cur.execute("INSERT INTO results VALUES (?, ?, ?)",
                    (match_id, winner_id, loser_id))
        cur.execute("INSERT INTO history VALUES (?, ?, ?)",
                    (match_id, str(match_result.moves), match_result.reason))
        cur.executemany("INSERT INTO times VALUES (?, ?, ?)",
                        [(match_id, winner_id, w_time),
                         (match_id, loser_id, l_time)])
        self.db.commit()

    def games(self, round_num=None):
        """Convert all matches in a specified round to displayable JSON """
        _round = self.round
        if _round == -1:
            return None
        elif not round_num:
            round_num = _round
        games = self._get_games(round_num)
        return [game2dict(*r) for r in games]

    def _finalize_round(self):
        # get finished matches from max(round) -- if none, do nothing
        match_results = list(map(ResultRecord._make, self._latest_results()))
        if not match_results or self.winner:
            return

        active_agents = list(map(AgentRecord._make, self._active_agents()))
        ratings = {a.id: a.rating for a in active_agents}
        times = {a.id: [] for a in active_agents}
        actuals = {a.id: 0 for a in active_agents}
        expected = {a.id: 0 for a in active_agents}

        for row in match_results:
            q_w = 10 ** (ratings[row.winner_id] / 400.)
            q_l = 10 ** (ratings[row.loser_id] / 400.)
            expected[row.loser_id] += q_l / (q_w + q_l)

            # Do not update the winner if the match was lost in the first
            # two moves
            num_moves = len(self._get_moves(row.id).fetchone()[0])
            if num_moves > 2:
                expected[row.winner_id] += q_w / (q_w + q_l)
                actuals[row.winner_id] += 1

            time_winner = self._get_time(row.id, row.winner_id).fetchone()[0]
            times[row.winner_id].append(time_winner / MOVE_TIME_LIMIT_MILLIS)
            time_loser = self._get_time(row.id, row.loser_id).fetchone()[0]
            times[row.loser_id].append(time_loser / MOVE_TIME_LIMIT_MILLIS)

        penalties = {a.id: len(times[a.id]) * a.loadtime / 2 + sum(times[a.id])
                     for a in active_agents}

        def diff(agent_id):
            diff = (actuals[agent_id] - expected[agent_id])
            return K * diff

        _ratings = sorted([(ratings[a.id] + diff(a.id), -penalties[a.id], a.id)
                           for a in active_agents])
        logger.debug(
            "Rankings for round {}: {}".format(
                self.round, _ratings
            )
        )

        # The number of players to eliminate is the greater of the number that
        # would result in the next round being a power of 2, or the number that
        # would leave only one player left if there are no more than 4 to start
        k = len(_ratings)
        power_of_two = int(2 ** int(math.ceil(math.log(k, 2)) - 1))
        if k > 4:
            N = k - power_of_two
        else:
            N = k - 1
        for idx, (rating, _, agent_id) in enumerate(_ratings):
            self.db.execute(
                "UPDATE agents SET rating=?, active=? WHERE rowid=?",
                (rating, idx >= N, agent_id)
            )
        logger.info(
            "Updated {} agents and eliminated {} in round {}.".format(
                len(active_agents), N, self.round
            )
        )
        self.db.commit()

    def _get_player_obj(self, agent_id):
        """Return a deserialized copy of an agent object from the db ID """
        cur = self.db.cursor()
        cur.execute("""
            SELECT rowid, pickle, rating, loadtime
            FROM agents WHERE rowid=?
            """, (agent_id,))
        rec = AgentRecord._make(cur.fetchone())
        with open(rec.pickle, 'rb') as pkl:
            player = Player(agent_id,
                            "Agent_{}".format(agent_id), pickle.load(pkl))
        return player

    def _make_matches(self):
        if self.winner:
            return
        agents = list(map(AgentRecord._make, self._active_agents()))
        if len(agents) <= 4:
            matches = roundrobin(agents, self.round + 1)
        else:
            matches = random_pairings(agents, self.round + 1)
        self._add_matches(matches)
        logger.info(
            "Adding {} matches for round {}"
            .format(len(matches), self.round)
        )

    def _add_matches(self, matches):
        self.db.executemany("""
            INSERT INTO matches(round, agentA, agentB)
            VALUES (?, ?, ?)
            """, matches)
        self.db.commit()

    def _active_agents(self):
        """Lazily select agents that have not been marked as eliminated
        """
        cur = self.db.cursor()
        cur.execute("""
            SELECT rowid, pickle, rating, loadtime
            FROM agents
            WHERE active=1
        """)
        return cur

    def _unfinished_matches(self):
        """Lazily select matches from the DB that have no recorded results """
        cur = self.db.cursor()
        cur.execute("""
            SELECT matches.*
            FROM matches
            LEFT JOIN results ON results.matchId=matches.rowid
            WHERE results.matchId IS NULL
        """)
        return cur

    def _latest_results(self):
        """Lazily select the match results from the most recent round """
        cur = self.db.cursor()
        cur.execute("""
            SELECT results.*
            FROM matches
            LEFT JOIN results ON results.matchId=matches.rowid
            WHERE matches.round=(SELECT MAX(matches.round) FROM matches)
        """)
        return cur

    def _get_games(self, round_num):
        """Lazily select the move history from the specified round """
        cur = self.db.cursor()
        cur.execute("""
            SELECT agentA, agentB, moves
            FROM matches
            LEFT JOIN history ON history.matchId=matches.rowid
            WHERE matches.round=?
        """, (round_num,))
        return cur

    def _get_time(self, match_id, agent_id):
        res = self.db.execute(
            "SELECT time FROM times WHERE matchId=? AND agentId=?",
            (match_id, agent_id)
        )
        return res

    def _get_moves(self, match_id):
        res = self.db.execute(
            "SELECT moves FROM history WHERE matchId=?",
            (match_id,)
        )
        return res

    def _get_game(self, match_id):
        res = self.db.execute("""
            SELECT matches.agentA, matches.agentB, history.moves
            FROM matches WHERE rowid=?
            LEFT JOIN history ON history.matchId=matches.rowid
            """, (match_id,)
        )
        return res


class MatchRunner:
    """Conduct a tournament by running matches concurrently with a threadpool
    """

    def __init__(self, num_processes, manager):
        self.pool = ThreadPool(processes=num_processes)
        self.manager = manager

    def _run_matches(self, matches):
        if not matches:
            return
        logger.info(
            "Running {} matches in round {}".format(
                len(matches), self.manager.round
            )
        )
        for output in self.pool.imap_unordered(_run, matches):
            if output is None:
                continue
            logger.debug(output)
            self.manager.record_match(*output)

    def run_tournament(self):
        while self.manager.winner is None:
            matches = self.manager.get_open_matches()
            self._run_matches(matches)

        final_four = self.manager.games()
        logger.debug(final_four)
        print(final_four)
        print("Overall winner:\n", self.manager.winner)
        logger.info("Overall Winner: {}".format(self.manager.winner))


def _run(args):
    player1, player2, match_id = args
    logger.info("Launching match {}".format(match_id))
    result = isolation.play(
        player1, player2, time_limit=MOVE_TIME_LIMIT_MILLIS,
        move_timeout=MOVE_TIMEOUT
    )
    logger.info("Completed match {}".format(match_id))
    return match_id, result


def _init_db(db_conn):
    logger.debug("Creating clean database")
    db_conn.executescript("""
        DROP TABLE IF EXISTS agents;
        DROP TABLE IF EXISTS matches;
        DROP TABLE IF EXISTS results;
        DROP TABLE IF EXISTS times;
        DROP TABLE IF EXISTS history;
        CREATE TABLE agents(
            rowid INTEGER PRIMARY KEY,
            pickle TEXT,
            rating REAL DEFAULT 1500.0,
            loadtime REAL,
            active INTEGER DEFAULT 1
        );
        CREATE TABLE matches(
            rowid INTEGER PRIMARY KEY,
            round INTEGER,
            agentA REFERENCES agents(rowid),
            agentB REFERENCES agents(rowid)
        );
        CREATE TABLE results(
            matchId REFERENCES matches(rowid),
            winnerId REFERENCES agents(rowid),
            loserId REFERENCES agents(rowid),
            PRIMARY KEY(matchId)
        );
        CREATE TABLE times(
            matchId REFERENCES matches(rowid),
            agentId REFERENCES agents(rowid),
            time REAL,
            PRIMARY KEY(matchId, agentId)
        );
        CREATE TABLE history(
            matchId REFERENCES matches(rowid),
            moves TEXT,
            reason TEXT
        );
        """)
    db_conn.commit()


def main(db_conn, src_folder, num_processes):
    manager = AgentManager(db_conn)
    runner = MatchRunner(num_processes, manager)
    runner.run_tournament()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument(
        "-i", "--input", default=os.getcwd(),
        help="Name of the input folder containing agent folders", dest='directory')
    parser.add_argument(
        "-p", "--processes", type=int, default=2,
        help="Number of processes to use for running matches")
    parser.add_argument(
        "-d", "--database", action="store", default="tournament.db",
        help="Name to use for sqlite database of results " +
             "(ignored if -t is used)")
    parser.add_argument(
        "-t", "--temp", action="store_true",
        help="Use an in-memory database (results not saved)")
    parser.add_argument(
        "-l", "--loglevel", default=Level.INFO.name,
        choices=[l.name for l in Level],
        help="")
    args = parser.parse_args()
    logger.setLevel(args.loglevel)
    logger.info("Command line arguments {!s}".format(vars(args)))

    db_name = ":memory:" if args.temp else args.database
    db_exists = os.path.isfile(args.database)
    logger.info("Starting database connection: {}".format(db_name))
    db_conn = sqlite3.connect(db_name)

    # if the database exists, the script will try to continue playing any
    # unfinished matches
    if args.temp or not db_exists:
        _init_db(db_conn)
        load_agents(db_conn, args.directory)

    curr = db_conn.cursor()
    curr.execute("SELECT count(*) FROM agents")
    num_agents = curr.fetchone()[0]
    if not num_agents:
        logger.error((
            "Unable to continue. No agents found in the database. Make " +
            "sure that all competition_agent.py files exist in subfolders " +
            "of {} and check the database in {}").format(
            os.path.abspath(args.directory), db_name))
        exit()
    logger.info("Initiating tournament with {} agents.".format(num_agents))

    main(db_conn, args.directory, args.processes)
