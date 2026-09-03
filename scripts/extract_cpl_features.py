"""
Multiprocess Stockfish centipawn-loss (CPL) extraction for EloSense v3 (W3).

Single-threaded POC (elosense_v3.ipynb section 7) measured 1.48s/game at depth 10.
Full dataset (66,879 games) would take ~27 hours single-threaded, and a 4-worker
smoke test showed multiprocess throughput well under linear scaling, so this
script samples instead of running the full dataset:
  - runs one Stockfish engine per worker process
  - samples from the player-disjoint test set (needed for an honest eval)
  - samples separately from the much larger train set

Caches raw per-move CPL lists per game so downstream notebook cells can derive
blunder rate, variance, and game-phase splits without re-running the engine.

Usage: python scripts/extract_cpl_features.py
"""
import io
import json
import time
from multiprocessing import Pool

import chess
import chess.engine
import chess.pgn
import pandas as pd

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
CPL_DEPTH = 10
CPL_CAP = 1000
N_WORKERS = 8
TEST_SAMPLE_SIZE = 4000
TRAIN_SAMPLE_SIZE = 4000
RANDOM_STATE = 42

DATA_PATH = "data/club_games_data.csv"
FEATURES_V3_PATH = "features/pgn_features_v3.csv"
OUT_PATH = "features/cpl_features_v3.csv"

_engine = None


def _init_worker():
    global _engine
    _engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)


def _game_white_cpls(pgn_text):
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return None
    board = game.board()
    cpls = []
    for ply, move in enumerate(game.mainline_moves()):
        if ply % 2 == 0:
            info_before = _engine.analyse(board, chess.engine.Limit(depth=CPL_DEPTH))
            score_before = info_before["score"].pov(chess.WHITE).score(mate_score=CPL_CAP)
            board.push(move)
            info_after = _engine.analyse(board, chess.engine.Limit(depth=CPL_DEPTH))
            score_after = info_after["score"].pov(chess.WHITE).score(mate_score=CPL_CAP)
            cpl = max(0, min(score_before, CPL_CAP) - max(score_after, -CPL_CAP))
            cpls.append(cpl)
        else:
            board.push(move)
    return cpls


def _process_row(args):
    row_id, pgn_text = args
    try:
        cpls = _game_white_cpls(pgn_text)
    except Exception:
        cpls = None
    return row_id, cpls


def main():
    df_full = pd.read_csv(DATA_PATH)
    feats = pd.read_csv(FEATURES_V3_PATH)
    assert len(df_full) == len(feats)

    test_idx_all = feats.index[feats["split"] == "test"].tolist()
    test_idx = (
        pd.Series(test_idx_all)
        .sample(n=min(TEST_SAMPLE_SIZE, len(test_idx_all)), random_state=RANDOM_STATE)
        .tolist()
    )
    train_idx_all = feats.index[feats["split"] == "train"].tolist()
    train_idx = (
        pd.Series(train_idx_all)
        .sample(n=min(TRAIN_SAMPLE_SIZE, len(train_idx_all)), random_state=RANDOM_STATE)
        .tolist()
    )
    target_idx = test_idx + train_idx
    print(f"scoring {len(target_idx)} games ({len(test_idx)} test, {len(train_idx)} train sample)")

    jobs = [(i, df_full.loc[i, "pgn"]) for i in target_idx]

    start = time.time()
    results = {}
    with Pool(N_WORKERS, initializer=_init_worker) as pool:
        for n_done, (row_id, cpls) in enumerate(pool.imap_unordered(_process_row, jobs), 1):
            results[row_id] = cpls
            if n_done % 500 == 0:
                elapsed = time.time() - start
                rate = n_done / elapsed
                eta_min = (len(jobs) - n_done) / rate / 60
                print(f"{n_done}/{len(jobs)} done, {rate:.2f} games/sec, eta {eta_min:.1f} min")

    elapsed = time.time() - start
    print(f"done: {len(jobs)} games in {elapsed:.1f}s ({len(jobs)/elapsed:.2f} games/sec)")

    rows = []
    for row_id in target_idx:
        cpls = results.get(row_id)
        rows.append({
            "row_id": row_id,
            "split": feats.loc[row_id, "split"],
            "n_moves_scored": len(cpls) if cpls else 0,
            "cpl_list": json.dumps(cpls) if cpls else None,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"saved {OUT_PATH}, shape {out_df.shape}")


if __name__ == "__main__":
    main()
