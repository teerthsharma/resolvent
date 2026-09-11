"""REMOVAL-ECHO v1, D2: NBA absences against the additive RAPM prediction.

Pre-registered at docs/prereg/REMOVAL_ECHO_v1.md (e238db6, SHA-256 pinned below):
  D2  RAPM residual variance on injury weeks vs placebo weeks
      prediction x1.5 with CI; counter x1.0 -- RAPM already absorbs rerouting.
  ECHO  lag of peak impulse response, expected at 1-3 games.

RAPM is additive by construction: a stint's margin is the sum of the ten players'
values. An absence is a dated REMOVAL; the rest of the roster re-equilibrates. If
the re-equilibration is not a sum of the same per-player values, the residual
against the additive prediction widens on the weeks a rotation player is out.

The test is FROZEN below (FROZEN, hashed as FROZEN_SHA) and was fixed before any
residual was computed. The instruments S1/S2 are removal_echo's, not rebuilt here.

DATA (hash-pinned, CC BY 4.0 / MIT, sportsdataverse-data GitHub releases, derived
from stats.nba.com; public URLs, no key): per-action on-court lineups, per-action
play-by-play scores, player box scores, inactive lists, team game logs.

RUN:  python -m ceqjepa.d2_nba --selfcheck          (offline, synthetic)
      python -m ceqjepa.d2_nba --cache DIR [--gate]  (real data, cached in DIR)
"""

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ceqjepa.removal_echo import REFUSED, bh, echo, placebo_rank

PREREG = Path(__file__).resolve().parents[1] / "docs" / "prereg" / "REMOVAL_ECHO_v1.md"
PREREG_SHA = "0e7591fea0d1995af7ac398d7810a9e5b854edd80e9a31c7cd6cb70307b05264"
SRC = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
FILES = {  # name: (release tag, SHA-256)
    "nba_lineups_2023.parquet": ("nba_stats_game_lineups", "3709347d4980e8e9a6a22b06e0c0add4e831649aaaa4ac6fbdf769940ddb5c1a"),
    "nba_lineups_2024.parquet": ("nba_stats_game_lineups", "07b7961f3f690e44bb652ce5da0742c9c2cfe100e5ff8ca87857addbae213c21"),
    "nba_play_by_play_2023.parquet": ("nba_stats_pbp", "4d7fb91e444f8b9630f2c3e94e7f32b09734efdc258a3b7e24437e02a38f0714"),
    "nba_play_by_play_2024.parquet": ("nba_stats_pbp", "615b74d5dd35fd1e5b55f8f2fc726fa2e2cabdeed57baedc66f0f13a878505df"),
    "player_boxscores_2023.parquet": ("nba_stats_player_boxscores", "3d3ba28b158dfe2fb0b2239318334eea014f2f008f465d3469576e4e308ecf98"),
    "player_boxscores_2024.parquet": ("nba_stats_player_boxscores", "2ed43e1d2b54a232e730a612c308aab9adc771990d1dcabafe59fdfa932af1fd"),
    "game_rosters_2023.parquet": ("nba_stats_game_rosters", "2b4ae945acb7c76c0848364a0c28b610ba5458963f2d0626ab437c742951f341"),
    "game_rosters_2024.parquet": ("nba_stats_game_rosters", "869237393dd304542ffb970fc3e032a70972347345017532b6d6a1427de47787"),
    "player_game_logs_2023.parquet": ("nba_stats_player_game_logs", "b5923c8cfcdcdc16913e9184e37182b70c3d85d370902fa84778c4d1243df711"),
    "player_game_logs_2024.parquet": ("nba_stats_player_game_logs", "38b2c2555f584c61c2da6e161970dd50aa73aeab62f9dd4909a116070d2b2a85"),
}

# ------------------------------------------------------------ THE FROZEN TEST ---
FROZEN = {
    "seasons": [2023, 2024],  # END-year asset keys: 2022-23 and 2023-24
    "games": "regular season only, game_id prefix 002",
    "rotation": "per team-season: >= 20 games on court for the team AND >= 20.0 mean box minutes in them",
    "out_game": "team game strictly between the player's first and last on-court game for that team, "
                "with no on-court action, and no box comment containing Coach's Decision",
    "removal_date": "first game of a maximal run of consecutive out-games (an absence spell)",
    "return_game": "first game after an absence spell",
    "week": "per team, calendar week starting Monday",
    "injury_week": "team-week containing >= 1 removal date of a rotation player",
    "placebo_week": "team-week containing no removal date, no return game, and no game 1-3 games after a "
                    "removal date, for any rotation player",
    "amendment": "placebo_week gained its lag 1-3 exclusion, and the selfcheck league grew to two 82-game "
                 "seasons, after the synthetic selfcheck showed post-removal spillover contaminating placebo "
                 "weeks (planted-positive fake VRs median 1.22, rank 43/200); then the matching moved from "
                 "pooled density strata to team-season x stratum cells after the selfcheck showed the pooled "
                 "placebo-in-time null at median 0.76 on an exactly additive league (singleton cells, from "
                 "injury-heavy team-seasons, carry r^2 168.8 vs 127.7 and never supply fakes). Both changes "
                 "were made on synthetic data only, before any real residual was computed",
    "density_strata": [[1, 2], [3], [4, 5]],
    "matching": "same team, same season, same density: within cells of team-season x games-in-week stratum, "
                "injury team-games in cells with no placebo team-game are dropped and placebo team-games are "
                "reweighted to the injury games' cell shares; placebo-in-time draws are made within the same cells",
    "rapm": "possession-level ridge per season; y = home pts - away pts in the possession; "
            "x = +1 home five, -1 away five at the possession's first action; unpenalised home intercept",
    "lambda_grid": [250, 500, 1000, 2000, 4000, 8000, 16000, 32000],
    "lambda_cv": "5-fold by game, minimise out-of-fold squared game-margin error",
    "residual": "team-perspective game margin minus the sum of its possession predictions",
    "statistic": "VR = weighted var(injury-week team-game residuals) / weighted var(placebo-week ones)",
    "interval": "95% percentile bootstrap resampling team-seasons",
    "bootstrap_B": 2000,
    "placebo_time_R": 999,
    "placebo_time_draw": "per team-season x stratum, min(n_injury_weeks, floor(n_placebo_weeks/2)) placebo "
                         "weeks relabelled fake-injury; fake VR against the remaining placebo weeks",
    "refit_without": "10-fold cross-fit by game; every training set excludes all games in an injury week of "
                     "either team; lambda from the full-season CV; every residual out-of-sample",
    "surrogate": "possession outcomes = full-fit prediction + possession residuals resampled iid; "
                 "identical pipeline including lambda CV",
    "surrogate_reps_vr": 20,
    "sc_events": "absence spells >= 4 games with no other >= 4-game spell start on that team in the 10 games before",
    "sc_window": [10, 4],
    "sc_donors": "teams with no >= 4-game spell start in the 14-game window, aligned by team game number",
    "sc_control": "same panel, a random donor as the treated unit",
    "echo_y": "squared residual, z-scored within team-season",
    "echo_max_lag": 10,
    "echo_null": "999 circular shifts of each team-season's removal series, offset uniform in [11, n-11]; "
                 "band = 95th percentile of max over lags of |mean h|",
    "score_rule": "prediction x1.5 holds if the 95% CI contains 1.5 and excludes 1.0; counter x1.0 holds if "
                  "the CI contains 1.0 and excludes 1.5; otherwise say where the CI sits; any effect claim "
                  "also needs placebo-in-time p <= 0.05",
    "secondary": "any-out weeks vs fully-healthy weeks, same statistic, reported and not scored",
    "primary": "the full-season in-sample fit (the league's own RAPM); the refit-without variant is "
               "reported beside it and scored by the same rule; synthetic control and echo use the full fit",
    "seed": 20260911,
}
FROZEN_SHA = hashlib.sha256(json.dumps(FROZEN, sort_keys=True).encode()).hexdigest()
STRATUM = {g: i for i, grp in enumerate(FROZEN["density_strata"]) for g in grp}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(cache):
    """Download any missing file, then refuse to go on unless every pin matches."""
    cache.mkdir(parents=True, exist_ok=True)
    for name, (tag, pin) in FILES.items():
        path = cache / name
        if not path.exists():
            urllib.request.urlretrieve(SRC + tag + "/" + name, path)
        got = sha256(path)
        if got != pin:
            raise SystemExit("HASH MISMATCH %s: %s != %s" % (name, got, pin))


# --------------------------------------------------------------- loading --------
HOME = ["home_player_%d" % i for i in range(1, 6)]
AWAY = ["away_player_%d" % i for i in range(1, 6)]


def load_season(cache, s):
    """One season as plain arrays: possessions, games, appearances, absence reasons."""
    L = pd.read_parquet(cache / ("nba_lineups_%d.parquet" % s))
    p = pd.read_parquet(cache / ("nba_play_by_play_%d.parquet" % s),
                        columns=["game_id", "action_number", "score_home", "score_away", "possession_number"])
    p["game_id"] = p.game_id.astype(str).str.zfill(10)
    assert (p.game_id.values == L.game_id.values).all(), "pbp and lineups rows do not align"
    assert (p.action_number.values == L.action_number.values).all(), "pbp and lineups rows do not align"
    keep = p.game_id.str[:3].eq("002").values
    p, L = p[keep].reset_index(drop=True), L[keep].reset_index(drop=True)
    for c in ("score_home", "score_away"):
        v = pd.to_numeric(p[c].replace("", np.nan))
        p[c] = v.groupby(p.game_id).ffill().fillna(0.0)
    d = (p.groupby("game_id").score_home.diff().fillna(p.score_home)
         - p.groupby("game_id").score_away.diff().fillna(p.score_away))
    inp = (p.possession_number > 0).values
    key = p.game_id[inp] + ":" + p.possession_number[inp].astype(int).astype(str)
    first = np.flatnonzero(inp)[~key.duplicated().values]
    y = d[inp].groupby(key.values, sort=False).sum().values
    T = pd.read_parquet(cache / ("player_game_logs_%d.parquet" % s))
    T = T[T.game_id.str[:3] == "002"]
    hm = T[T.matchup.str.contains(" vs. ")].set_index("game_id")
    aw = T[T.matchup.str.contains(" @ ")].set_index("game_id")
    games = pd.DataFrame({"game_id": hm.index, "date": pd.to_datetime(hm.game_date.values),
                          "home": hm.team_id.values, "away": aw.team_id.reindex(hm.index).values})
    gix = pd.Series(np.arange(len(games)), index=games.game_id)
    B = pd.read_parquet(cache / ("player_boxscores_%d.parquet" % s))
    B = B[B.game_id.str[:3] == "002"]
    mm = B.minutes.fillna("").str.split(":")
    B = B.assign(min=pd.to_numeric(mm.str[0], errors="coerce").fillna(0)
                 + pd.to_numeric(mm.str[1], errors="coerce").fillna(0) / 60)
    on = pd.DataFrame({"game_id": np.repeat(L.game_id.values, 10),
                       "pid": L[HOME + AWAY].values.ravel()}).drop_duplicates()
    app = on.merge(B[["game_id", "person_id", "team_id", "min"]].rename(columns={"person_id": "pid"}),
                   on=["game_id", "pid"])
    com = B.comment.fillna("")
    cd = com.str.contains("Coach's Decision")
    R = pd.read_parquet(cache / ("game_rosters_%d.parquet" % s))
    return {"season": s, "games": games,
            "g": gix[p.game_id.values[first]].values,
            "H": L[HOME].values[first], "A": L[AWAY].values[first], "y": y.astype(float),
            "app": app, "n_on_court": len(on),
            "coach": set(zip(B.game_id[cd], B.person_id[cd])),
            "reason": dict(zip(zip(B.game_id, B.person_id), com)),
            "inactive": set(zip(R.game_id, R.player_id))}
# ---------------------------------------------------------- absences, weeks ------
def label(S):
    """Team-game table with removal/return/out flags and the frozen week labels.

    Uses only who was on court, box minutes and box comments -- never a residual.
    """
    G = S["games"]
    n = len(G)
    tg = pd.concat([pd.DataFrame({"team": G.home.values, "gi": np.arange(n), "sign": 1.0, "date": G.date.values}),
                    pd.DataFrame({"team": G.away.values, "gi": np.arange(n), "sign": -1.0, "date": G.date.values})])
    tg = tg.sort_values(["team", "date", "gi"]).reset_index(drop=True)
    tg["k"] = tg.groupby("team").cumcount()
    tg["week"] = tg.date - pd.to_timedelta(tg.date.dt.weekday, unit="D")
    gid = G.game_id.values
    app = S["app"].merge(pd.DataFrame({"team_id": tg.team, "game_id": gid[tg.gi.values], "k": tg.k}),
                         on=["team_id", "game_id"])
    agg = app.groupby(["team_id", "pid"]).agg(n=("k", "size"), mpg=("min", "mean"),
                                              k0=("k", "min"), k1=("k", "max")).reset_index()
    rot = agg[(agg.n >= 20) & (agg.mpg >= 20.0)]
    on = set(zip(app.team_id, app.pid, app.k))
    kg = dict(zip(zip(tg.team, tg.k), tg.gi))
    out, removal, ret, spells = {}, set(), set(), []
    for r in rot.itertuples(index=False):
        run, start = 0, 0
        for k in range(r.k0, r.k1 + 1):
            if (r.team_id, r.pid, k) not in on and (gid[kg[(r.team_id, k)]], r.pid) not in S["coach"]:
                out[(r.team_id, k)] = out.get((r.team_id, k), 0) + 1
                start = k if run == 0 else start
                run += 1
            elif run:  # k1 is an appearance, so no spell is left open
                spells.append((r.team_id, r.pid, start, run))
                removal.add((r.team_id, start))
                ret.add((r.team_id, k))
                run = 0
    key = list(zip(tg.team, tg.k))
    post = {(t, k0 + j) for t, k0 in removal for j in (1, 2, 3)}
    tg["removal"] = [x in removal for x in key]
    tg["ret"] = [x in ret for x in key]
    tg["post"] = [x in post for x in key]
    tg["n_out"] = [out.get(x, 0) for x in key]
    wk = tg.groupby(["team", "week"])
    tg["n_wk"] = wk.gi.transform("size").clip(1, 5)
    tg["inj"] = wk.removal.transform("any")
    tg["plc"] = ~(tg.inj | wk.ret.transform("any") | wk.post.transform("any"))
    tg["anyout"] = wk.n_out.transform("sum") > 0
    tg["st"] = tg.n_wk.map(STRATUM).astype(int)
    tg["ts"] = S["season"] * 10_000_000_000 + tg.team.astype(np.int64)
    tg["wid"] = pd.factorize(tg.ts.astype(str) + tg.week.astype(str))[0]
    tg["season"] = S["season"]
    why = {"inactive list": 0, "injury/illness comment": 0, "other comment": 0, "no box row": 0}
    for t, pid, k0, _ in spells:
        gk = gid[kg[(t, k0)]]
        c = S["reason"].get((gk, pid))
        why["inactive list" if (gk, pid) in S["inactive"] else
            "no box row" if c is None else
            "injury/illness comment" if "Injury" in c else "other comment"] += 1
    return tg, spells, rot, why
# ------------------------------------------------------------------ RAPM ---------
def design(S):
    """Sparse possession design (+1 home five, -1 away five, home intercept) and its
    game-level sum Z, so a game's additive prediction is Z @ beta."""
    n = len(S["y"])
    _, inv = np.unique(np.concatenate([S["H"].ravel(), S["A"].ravel()]), return_inverse=True)
    P = int(inv.max()) + 1
    cols = np.hstack([inv[: 5 * n].reshape(n, 5), inv[5 * n:].reshape(n, 5), np.full((n, 1), P)]).ravel()
    X = sp.csr_matrix((np.tile(np.r_[np.ones(5), -np.ones(5), 1.0], n), (np.repeat(np.arange(n), 11), cols)),
                      shape=(n, P + 1))
    ng = len(S["games"])
    Gm = sp.csr_matrix((np.ones(n), (S["g"], np.arange(n))), shape=(ng, n))
    return X, (Gm @ X).toarray()


def gram(X, y, rows):
    Xr = X[rows]
    return (Xr.T @ Xr).toarray(), Xr.T @ y[rows]


def ridge(XtX, Xty, lam):
    d = np.full(XtX.shape[0], float(lam))
    d[-1] = 0.0  # the home intercept is not shrunk
    return np.linalg.solve(XtX + np.diag(d), Xty)


def cv_lambda(X, Z, y, g, rng):
    ng = Z.shape[0]
    fold = rng.permutation(ng) % 5
    XtX, Xty = gram(X, y, slice(None))
    margin = np.bincount(g, y, ng)
    grid = FROZEN["lambda_grid"]
    err = np.zeros(len(grid))
    for f in range(5):
        A, b = gram(X, y, fold[g] == f)
        te = fold == f
        for i, lam in enumerate(grid):
            err[i] += np.sum((margin[te] - Z[te] @ ridge(XtX - A, Xty - b, lam)) ** 2)
    return grid[int(np.argmin(err))], XtX, Xty


def residuals(S, y, tg_s, rng, out_variant=True):
    """Home-perspective game residuals: full-season fit, and the cross-fit that never
    trains on an injury-week game. Returns (r_in, r_out, lam, beta)."""
    X, Z = S["X"], S["Z"]
    lam, XtX, Xty = cv_lambda(X, Z, y, S["g"], rng)
    beta = ridge(XtX, Xty, lam)
    ng = Z.shape[0]
    margin = np.bincount(S["g"], y, ng)
    r_in = margin - Z @ beta
    r_out = None
    if out_variant:
        excl = np.zeros(ng, bool)
        excl[tg_s.gi.values[tg_s.inj.values]] = True
        fold = rng.permutation(ng) % 10
        r_out = np.empty(ng)
        for f in range(10):
            train = (fold != f) & ~excl
            A, b = gram(X, y, train[S["g"]])
            te = fold == f
            r_out[te] = margin[te] - Z[te] @ ridge(A, b, lam)
    return r_in, r_out, lam, beta


def surrogate_y(S, y, beta, rng):
    """Purely additive outcomes at the data's own possession-level residual noise.
    AS FROZEN, and flawed: see surrogate_y_game."""
    fit = S["X"] @ beta
    return fit + rng.choice(y - fit, size=len(y), replace=True)


def surrogate_y_game(S, y, beta, rng):
    """CORRECTION, added after the real run. y is home pts minus away pts, so a home
    possession's residual sits near +1.1 and an away one near -1.1; surrogate_y's iid
    resampling scrambles that alternation and put the game noise at sd 22.74 against the
    data's 12.62. Here each game keeps its own within-game residual pattern but its total
    is replaced by a draw from the pool of real game residuals: the additive prediction
    plus noise at the data's own game-level residual."""
    fit = S["X"] @ beta
    e = y - fit
    ng = len(S["games"])
    n, E = np.bincount(S["g"], minlength=ng), np.bincount(S["g"], e, ng)
    return fit + e - ((E - rng.choice(E, size=ng, replace=True)) / n)[S["g"]]
# ------------------------------------------------------------ statistics ---------
def wvar(r, w):
    m = np.sum(w * r) / np.sum(w)
    return np.sum(w * (r - m) ** 2) / np.sum(w)


def vr(r, cell, a, b):
    """var(r[a]) / var(r[b]) matched within cells (team-season x games-in-week stratum):
    a-games in cells with no b-game are dropped, b-games reweighted to a's cell shares."""
    m = int(cell.max()) + 1
    na, nb = np.bincount(cell[a], minlength=m), np.bincount(cell[b], minlength=m)
    both = (na > 0) & (nb > 0)
    return (wvar(r[a], both[cell[a]].astype(float))
            / wvar(r[b], np.divide(na, nb, out=np.zeros(m), where=both)[cell[b]]))


def boot_ci(r, st, a, b, ts, B, rng):
    """95% percentile interval, resampling team-seasons (the clusters)."""
    codes, uniq = pd.factorize(ts)
    members = [np.flatnonzero(codes == c) for c in range(len(uniq))]
    out = np.empty(B)
    for i in range(B):
        idx = np.concatenate([members[c] for c in rng.integers(len(members), size=len(members))])
        out[i] = vr(r[idx], st[idx], a[idx], b[idx])
    return np.percentile(out, [2.5, 97.5])


def placebo_time(r, st, a, b, wid, ts, R, rng):
    """Fake removal weeks drawn from placebo weeks, per team-season x stratum.
    Returns (rank, R + 1, fakes); rank 1 = the real VR exceeds every fake."""
    W = pd.DataFrame({"wid": wid, "ts": ts, "st": st, "a": a, "b": b}).drop_duplicates("wid")
    groups = []
    for _, grp in W.groupby(["ts", "st"]):
        pw = grp.wid.values[grp.b.values]
        k = min(int(grp.a.sum()), len(pw) // 2)
        if k:
            groups.append((pw, k))
    pb = np.flatnonzero(b)
    wb, rb, sb = wid[pb], r[pb], st[pb]
    fakes = np.empty(R)
    for i in range(R):
        f = np.isin(wb, np.concatenate([rng.choice(pw, k, replace=False) for pw, k in groups]))
        fakes[i] = vr(rb, sb, f, ~f)
    return 1 + int(np.sum(fakes >= vr(r, st, a, b))), R + 1, fakes


def sc_space(tg, spells_by_season, rng):
    """Placebo-in-space. Per long absence, removal_echo.placebo_rank on the team's
    residual series against donors that lost nobody for long; and the same panel with
    a donor -- a team whose players did not miss time -- as the fake treated unit.
    Rows: (treated rank, control rank, n units)."""
    pre, post = FROZEN["sc_window"]
    rows = []
    for s, spells in spells_by_season.items():
        t_s = tg[tg.season == s].sort_values("k")
        R = {t: g.r.values for t, g in t_s.groupby("team")}
        starts = {}
        for t, _, k0, n in spells:
            if n >= 4:
                starts.setdefault(t, set()).add(k0)
        for t in sorted(starts):
            for k0 in sorted(starts[t]):
                if k0 < pre or k0 + post > len(R[t]) or any(k0 - pre <= k < k0 for k in starts[t]):
                    continue
                donors = [d for d in R if d != t and len(R[d]) >= k0 + post
                          and not any(k0 - pre <= k < k0 + post for k in starts.get(d, ()))]
                if len(donors) < 2:
                    continue
                Y = np.column_stack([R[u][k0 - pre:k0 + post] for u in [t] + donors])
                a = placebo_rank(Y, 0, pre)
                c = placebo_rank(Y, 1 + int(rng.integers(len(donors))), pre)
                if a is not REFUSED and c is not REFUSED:
                    rows.append((a[0], c[0], a[1]))
    return np.array(rows, dtype=float).reshape(-1, 3)


def echo_curve(tg, col, R, rng):
    """Mean over team-seasons of removal_echo.echo(removal, y, 0..10) and the
    familywise null band from circular shifts of each removal series."""
    L = FROZEN["echo_max_lag"]
    ser = [(g.removal.values.astype(float), g[col].values)
           for _, g in tg.sort_values(["ts", "k"]).groupby("ts")]
    ser = [(m, y) for m, y in ser if m.std() > 0 and len(m) > 22]
    h = np.mean([echo(m, y, L) for m, y in ser], axis=0)
    null = np.array([np.mean([echo(np.roll(m, int(rng.integers(11, len(m) - 10))), y, L) for m, y in ser], axis=0)
                     for _ in range(R)])
    return h, float(np.percentile(np.abs(null).max(axis=1), 95)), null
# -------------------------------------------------------------- pipeline ---------
def prep(S):
    S["X"], S["Z"] = design(S)
    S["tg"], S["spells"], S["rot"], S["why"] = label(S)
    return S


def analyse(seasons, ys, rng, B, R, full=True):
    """The identical pipeline for real, surrogate and synthetic outcomes."""
    parts, fits = [], {}
    for S in seasons:
        s = S["season"]
        r_in, r_out, lam, beta = residuals(S, ys[s], S["tg"], rng, out_variant=full)
        t = S["tg"].copy()
        t["r"] = r_in[t.gi.values] * t.sign.values
        if full:
            t["r_out"] = r_out[t.gi.values] * t.sign.values
        parts.append(t)
        fits[s] = (lam, beta)
    tg = pd.concat(parts, ignore_index=True)
    st = pd.factorize(tg.ts.astype(str) + "_" + tg.st.astype(str))[0]  # matching cells: team-season x stratum
    a, b, ts, wid = tg.inj.values, tg.plc.values, tg.ts.values, tg.wid.values
    res = {"tg": tg, "fits": fits, "vr_in": vr(tg.r.values, st, a, b)}
    if not full:
        return res
    res["ci_in"] = boot_ci(tg.r.values, st, a, b, ts, B, rng)
    res["vr_out"] = vr(tg.r_out.values, st, a, b)
    res["ci_out"] = boot_ci(tg.r_out.values, st, a, b, ts, B, rng)
    res["pt_in"] = placebo_time(tg.r.values, st, a, b, wid, ts, R, rng)
    res["pt_out"] = placebo_time(tg.r_out.values, st, a, b, wid, ts, R, rng)
    ao = tg.anyout.values
    res["vr_sec"] = vr(tg.r.values, st, ao, ~ao)
    res["ci_sec"] = boot_ci(tg.r.values, st, ao, ~ao, ts, B, rng)
    res["sc"] = sc_space(tg, {S["season"]: S["spells"] for S in seasons}, rng)
    z = lambda x: (x - x.mean()) / x.std()
    tg["r2z"] = tg.groupby("ts").r.transform(lambda x: z(x ** 2))
    tg["rz"] = tg.groupby("ts").r.transform(z)
    res["echo"] = echo_curve(tg, "r2z", R, rng)
    res["echo_signed"] = echo_curve(tg, "rz", R, rng)
    return res


def score(ci):
    lo, hi = ci
    if lo <= 1.0 <= hi and hi < 1.5:
        return "COUNTER x1.0 HOLDS (CI contains 1.0, excludes 1.5)"
    if lo <= 1.5 <= hi and lo > 1.0:
        return "PREDICTION x1.5 HOLDS (CI contains 1.5, excludes 1.0)"
    if lo <= 1.0 and hi >= 1.5:
        return "INCONCLUSIVE (CI contains both 1.0 and 1.5)"
    return "NEITHER (CI sits %s)" % ("strictly between 1.0 and 1.5" if 1.0 < lo and hi < 1.5
                                     else "above 1.5" if lo > 1.5 else "below 1.0")


def sc_summary(rows):
    if not len(rows):
        return "no synthetic-control events"
    u, c = (rows[:, 0] - 1) / (rows[:, 2] - 1), (rows[:, 1] - 1) / (rows[:, 2] - 1)
    half = lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    return ("%d events, %.1f units each on average. Normalised rank (0 = largest post/pre RMSPE, "
            "uniform null mean 0.5): TREATED %.3f +/- %.3f; CONTROL (a donor that lost nobody) %.3f +/- %.3f. "
            "Treated at rank 1 in %.3f of events (uniform %.3f). BH q=0.05 discoveries on p = rank/n: %d "
            "(smallest achievable p %.3f)" % (len(rows), rows[:, 2].mean(), u.mean(), half(u), c.mean(), half(c),
                                              (rows[:, 0] == 1).mean(), (1 / rows[:, 2]).mean(),
                                              int(bh(rows[:, 0] / rows[:, 2]).sum()), (1 / rows[:, 2]).min()))


def echo_line(e):
    h, band, _ = e
    k = int(np.argmax(np.abs(h)))
    return ("h(0..10) = [%s]; peak |h| = %.4f at lag %d; familywise null band %.4f -> %s"
            % (", ".join("%+.4f" % v for v in h), abs(h[k]), k, band,
               ("ECHO at lag %d" % k) if abs(h[k]) > band else "NO ECHO (peak inside the null band)"))


def report(tag, res):
    tg = res["tg"]
    a, b = tg.inj.values, tg.plc.values
    print("  [%s] team-games: injury-week %d, placebo-week %d; lambda %s; residual sd %.2f"
          % (tag, a.sum(), b.sum(), {s: f[0] for s, f in res["fits"].items()}, tg.r.std()))
    for v in ("in", "out"):
        rk, n, fk = res["pt_" + v]
        print("  [%s] %s VR = %.3f, 95%% CI [%.3f, %.3f]; placebo-in-time rank %d of %d (p = %.3f), "
              "fake VRs median %.3f, 95%% range [%.3f, %.3f]"
              % (tag, "FULL-FIT " if v == "in" else "REFIT-WITHOUT", res["vr_" + v], *res["ci_" + v],
                 rk, n, rk / n, np.median(fk), *np.percentile(fk, [2.5, 97.5])))
    print("  [%s] secondary any-out vs healthy VR = %.3f, 95%% CI [%.3f, %.3f]" % (tag, res["vr_sec"], *res["ci_sec"]))
    print("  [%s] placebo-in-space: %s" % (tag, sc_summary(res["sc"])))
    print("  [%s] echo, squared residual: %s" % (tag, echo_line(res["echo"])))
    print("  [%s] echo, signed residual:  %s" % (tag, echo_line(res["echo_signed"])))
# ------------------------------------------------------------- selfcheck ---------
def synthetic(rng, n_teams=30, n_days=60, season=2023):
    """A league whose margins are EXACTLY additive in per-player values, with Markov
    absences. Returns the dict load_season returns."""
    n_pl = 12
    beta = rng.normal(0.0, 0.03, (n_teams, n_pl))
    w = np.r_[np.full(5, 3.0), np.full(4, 1.5), np.full(3, 0.2)]
    out = np.zeros((n_teams, n_pl), bool)
    games, g, H, A, y, app, coach, inactive = [], [], [], [], [], [], set(), set()
    for day in range(n_days):
        out = np.where(out, rng.random(out.shape) < 0.6, rng.random(out.shape) < 0.04)
        out[:, 9:] = False
        perm = rng.permutation(n_teams)
        for ht, at in zip(perm[::2], perm[1::2]):
            gi, gid = len(games), "S%05d" % len(games)
            games.append((gid, pd.Timestamp("2022-10-17") + pd.Timedelta(days=2 * day), ht, at))
            fives = {}
            for t in (ht, at):
                avail = np.flatnonzero(~out[t])
                fives[t] = [rng.choice(avail, 5, replace=False, p=w[avail] / w[avail].sum()) for _ in range(4)]
                cnt = np.bincount(np.concatenate(fives[t]), minlength=n_pl)
                for j in range(n_pl):
                    if cnt[j]:
                        app.append((gid, t * 100 + j, t, 12.0 * cnt[j]))
                    elif out[t, j]:
                        inactive.add((gid, t * 100 + j))
                    else:
                        coach.add((gid, t * 100 + j))  # dressed, not used: not a removal
            for q in range(4):
                hf, af = fives[ht][q], fives[at][q]
                y.extend(beta[ht, hf].sum() - beta[at, af].sum() + 0.015 + rng.normal(0.0, 1.2, 25))
                g.extend([gi] * 25)
                H.extend([ht * 100 + hf] * 25)
                A.extend([at * 100 + af] * 25)
    G = pd.DataFrame(games, columns=["game_id", "date", "home", "away"])
    return {"season": season, "games": G, "g": np.array(g), "H": np.array(H), "A": np.array(A),
            "y": np.array(y), "app": pd.DataFrame(app, columns=["game_id", "pid", "team_id", "min"]),
            "n_on_court": len(app), "coach": coach, "reason": {}, "inactive": inactive}


def plant(S, y, lag, sd, rng):
    """Non-additive shock on the team's margin `lag` games after each removal date."""
    y = y.copy()
    tg = S["tg"]
    gi_at = dict(zip(zip(tg.team, tg.k), zip(tg.gi, tg.sign)))
    npos = np.bincount(S["g"], minlength=len(S["games"]))
    for t, k in zip(tg.team[tg.removal], tg.k[tg.removal]):
        if (t, k + lag) in gi_at:
            gi, sign = gi_at[(t, k + lag)]
            y[S["g"] == gi] += sign * rng.normal(0.0, sd) / npos[gi]
    return y


def selfcheck():
    rng = np.random.default_rng(7)
    seasons = [prep(synthetic(rng, n_days=82, season=s)) for s in (2023, 2024)]
    print("synthetic league, two seasons: %d games, %d possessions, %d rotation players, %d absence spells"
          % tuple(sum(v) for v in zip(*[(len(S["games"]), len(S["y"]), len(S["rot"]), len(S["spells"]))
                                        for S in seasons])))
    print("(1) PLANTED NEGATIVE: margins exactly additive. The pipeline must read x1.0.")
    neg = analyse(seasons, {S["season"]: S["y"] for S in seasons}, rng, B=300, R=199)
    report("additive", neg)
    lo, hi = neg["ci_in"]
    assert lo <= 1.0 <= hi, "an exactly additive league reads non-additive"
    assert neg["pt_in"][0] / neg["pt_in"][1] > 0.05, "placebo-in-time calls an additive league significant"
    print("(2) PLANTED POSITIVE: a sd-25 non-additive shock 2 games after every removal (sized to ~x1.5).")
    pos = analyse(seasons, {S["season"]: plant(S, S["y"], 2, 25.0, rng) for S in seasons}, rng, B=300, R=199)
    report("planted", pos)
    assert pos["ci_in"][0] > 1.0, "a planted non-additive shock is not seen"
    assert pos["pt_in"][0] / pos["pt_in"][1] <= 0.05, "a planted shock does not out-rank its placebos"
    h, band, _ = pos["echo"]
    assert int(np.argmax(np.abs(h))) == 2 and abs(h[2]) > band, "the echo does not peak at the planted lag 2"
    # A single-seed "no echo" assert would false-alarm 5% of the time by design; the band's
    # calibration was measured instead (1 of 20 additive leagues exceeded it, nominal 1).
    assert np.abs(neg["echo"][0]).max() < abs(h[2]) / 2, "an additive league echoes half as strongly as the plant"
    print("ALL SELF-CHECKS PASSED")


# -------------------------------------------------------------------- real --------
def run(cache, gate_only):
    got = sha256(PREREG)
    print("PREREG %s sha256 %s -> %s" % (PREREG.name, got, "MATCH" if got == PREREG_SHA else "MISMATCH"))
    if got != PREREG_SHA:
        raise SystemExit("STOP: the pre-registration does not match its pinned hash")
    fetch(cache)
    print("FILES (all SHA-256 pins verified):")
    for name, (tag, pin) in FILES.items():
        print("  %-32s %10d B  %s  %s" % (name, (cache / name).stat().st_size, pin, SRC + tag + "/" + name))
    seasons = [prep(load_season(cache, s)) for s in FROZEN["seasons"]]
    print("GATE (counts only, no residuals):")
    ok = True
    for S in seasons:
        w = S["tg"].drop_duplicates("wid")
        n_inj, n_plc = int(w.inj.sum()), int(w.plc.sum())
        a_ok = len(np.unique(S["g"])) == len(S["games"]) and S["n_on_court"] - len(S["app"]) <= 5
        c_ok = n_inj >= 100 and n_plc >= 100
        ok &= a_ok and len(S["spells"]) > 0 and c_ok
        print("  %d: (a) %d games, all with possessions: %s; %d possessions; on-court players without a box row: %d"
              % (S["season"], len(S["games"]), a_ok, len(S["y"]), S["n_on_court"] - len(S["app"])))
        print("        (b) %d rotation players, %d absence spells; first missed game recorded as %s"
              % (len(S["rot"]), len(S["spells"]), S["why"]))
        print("        (c) %d team-weeks: %d injury, %d placebo, %d any-out, %d fully healthy; placebo weeks "
              "per team min %d" % (len(w), n_inj, n_plc, int(w.anyout.sum()), int((~w.anyout).sum()),
                                   int(w[w.plc].groupby("team").size().reindex(w.team.unique(), fill_value=0).min())))
    print("GATE VERDICT: %s" % ("PASS" if ok else "FAIL"))
    if gate_only or not ok:
        return
    print("FROZEN TEST (sha256 of the canonical JSON %s):" % FROZEN_SHA)
    for k in sorted(FROZEN):
        print("  %s: %s" % (k, FROZEN[k]))
    rng = np.random.default_rng(FROZEN["seed"])
    B, R = FROZEN["bootstrap_B"], FROZEN["placebo_time_R"]
    real = analyse(seasons, {S["season"]: S["y"] for S in seasons}, rng, B, R)
    sy = lambda: {S["season"]: surrogate_y(S, S["y"], real["fits"][S["season"]][1], rng) for S in seasons}
    sur = analyse(seasons, sy(), rng, B, R)
    reps = [sur["vr_in"]] + [analyse(seasons, sy(), rng, B, R, full=False)["vr_in"]
                             for _ in range(FROZEN["surrogate_reps_vr"] - 1)]
    print("MATCHED-NOISE ADDITIVE SURROGATE (identical pipeline; must read x1.0):")
    report("surrogate", sur)
    print("  [surrogate] %d replicates of the full-fit VR: mean %.3f, min %.3f, max %.3f -> %s"
          % (len(reps), np.mean(reps), np.min(reps), np.max(reps),
             "FIRES: reads x1.0" if sur["ci_in"][0] <= 1.0 <= sur["ci_in"][1] else "FAILS: the pipeline manufactures an effect"))
    print("REAL DATA:")
    report("real", real)
    print("SCORE (pre-registered: prediction x1.5, counter x1.0):")
    print("  full-fit RAPM (primary):  VR %.3f [%.3f, %.3f] -> %s; placebo-in-time p = %.3f"
          % (real["vr_in"], *real["ci_in"], score(real["ci_in"]), real["pt_in"][0] / real["pt_in"][1]))
    print("  refit without injury games: VR %.3f [%.3f, %.3f] -> %s; placebo-in-time p = %.3f"
          % (real["vr_out"], *real["ci_out"], score(real["ci_out"]), real["pt_out"][0] / real["pt_out"][1]))
    # Appended after every frozen computation, so the rng stream -- and every number above -- is unchanged.
    sg = lambda: {S["season"]: surrogate_y_game(S, S["y"], real["fits"][S["season"]][1], rng) for S in seasons}
    sur2 = analyse(seasons, sg(), rng, B, R)
    reps2 = [sur2["vr_in"]] + [analyse(seasons, sg(), rng, B, R, full=False)["vr_in"]
                               for _ in range(FROZEN["surrogate_reps_vr"] - 1)]
    print("CORRECTION (added after the real run): the frozen surrogate's residual sd was %.2f against the data's "
          "%.2f -- NOT matched noise. Re-run at the data's own game-level residual:"
          % (sur["tg"].r.std(), real["tg"].r.std()))
    report("surrogate-matched", sur2)
    print("  [surrogate-matched] residual sd %.2f vs data %.2f; %d replicates of the full-fit VR: mean %.3f, "
          "min %.3f, max %.3f -> %s"
          % (sur2["tg"].r.std(), real["tg"].r.std(), len(reps2), np.mean(reps2), np.min(reps2), np.max(reps2),
             "FIRES: reads x1.0" if sur2["ci_in"][0] <= 1.0 <= sur2["ci_in"][1] else "FAILS: the pipeline manufactures an effect"))


# ------------------------------------------------------- bound admission ---------
def admission(cache, n_perm=200, seed=20260911):
    """BOUND ADMISSION -- the pre-registration's section-3 gate, made able to fail.

    The first version of this measurement was an inline `python -c` that printed the
    rotation's minutes change when a teammate is out, with a confidence interval, and
    asserted nothing. The Inspector struck it: a check that cannot fail is not a gate,
    and admission IS a gate -- "a domain without re-equilibration is refused, not
    tested". The numbers it printed reproduced exactly (+1.81 +/- 0.38 minutes); what
    it lacked was any way to say no. This version has a pass bar and a planted negative.

    RE-EQUILIBRATION: when a rotation teammate is out, do the remaining rotation
    players' minutes rise, against their own full-strength games?
      PASS BAR   the per-player minutes change has a 95% CI excluding 0, AND it beats a
                 within-team permutation null at p <= 0.05.
      PLANTED    permuting WHICH games count as 'teammate out', within each team,
      NEGATIVE   destroys any link between an absence and teammate minutes by
                 construction. The gate must REFUSE that noise. If it admits permuted
                 noise in more than 10% of shuffles it is too loose, and the verdict is
                 void.
    Runs on its own path, never inside run(): run() draws the frozen headline from one
    seeded stream, and inserting anything there would change the headline.
    """
    import pandas as pd
    got = sha256(PREREG)
    if got != PREREG_SHA:
        raise SystemExit("STOP: the pre-registration does not match its pinned hash")
    fetch(cache)
    rng = np.random.default_rng(seed)
    verdicts = {}
    for s in FROZEN["seasons"]:
        S = load_season(cache, s)
        tg, spells, rot, why = label(S)
        B = pd.read_parquet(cache / ("player_boxscores_%d.parquet" % s))
        B = B[B.game_id.str[:3] == "002"].copy()
        mm = B.minutes.fillna("").str.split(":")
        B["mn"] = (pd.to_numeric(mm.str[0], errors="coerce").fillna(0)
                   + pd.to_numeric(mm.str[1], errors="coerce").fillna(0) / 60)
        tg = tg.copy()
        tg["game_id"] = S["games"].game_id.values[tg.gi.values]
        R = rot[["team_id", "pid"]].rename(columns={"pid": "person_id"})
        base = B.merge(R, on=["team_id", "person_id"])
        base = base[base.mn > 0][["team_id", "person_id", "game_id", "mn"]]
        g = tg[["team", "game_id", "n_out"]].rename(columns={"team": "team_id"})

        def dm_of(gtab):
            M = base.merge(gtab, on=["team_id", "game_id"])
            f = M[M.n_out == 0].groupby(["team_id", "person_id"]).mn.mean()
            o = M[M.n_out > 0].groupby(["team_id", "person_id"]).mn.mean()
            j = pd.concat([f.rename("f"), o.rename("o")], axis=1).dropna()
            return (j.o - j.f).values

        def ci(x):
            h = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
            return float(x.mean()), float(x.mean() - h), float(x.mean() + h)

        real = dm_of(g)
        m, lo, hi = ci(real)
        perm_means, perm_admit = [], 0
        for _ in range(n_perm):
            gp = g.copy()
            gp["n_out"] = gp.groupby("team_id").n_out.transform(lambda v: rng.permutation(v.values))
            pm, plo, _ = ci(dm_of(gp))
            perm_means.append(pm)
            perm_admit += plo > 0
        perm_means = np.asarray(perm_means)
        p = float((1 + np.sum(perm_means >= m)) / (1 + n_perm))
        admit_rate = perm_admit / n_perm
        admitted = lo > 0 and p <= 0.05
        verdicts[s] = admitted
        print("ADMISSION %d: %d rotation players with both kinds of game" % (s, len(real)))
        print("  REAL     minutes, teammate out vs full strength: %+.2f  95%% CI [%+.2f, %+.2f]" % (m, lo, hi))
        print("  PLANTED  %d within-team shuffles of 'teammate out': mean %+.3f, sd %.3f"
              % (n_perm, perm_means.mean(), perm_means.std(ddof=1)))
        print("           the gate ADMITS shuffled noise in %.3f of shuffles (must be <= 0.10)" % admit_rate)
        print("  permutation p = %.4f  (smallest achievable %.4f)" % (p, 1.0 / (1 + n_perm)))
        assert admit_rate <= 0.10, "the admission gate admits shuffled noise: too loose to decide anything"
        print("  VERDICT: %s" % ("ADMITTED -- the roster re-equilibrates" if admitted
                                 else "REFUSED -- no measurable re-equilibration"))
    return verdicts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selfcheck", action="store_true", help="offline synthetic must-fires")
    ap.add_argument("--gate", action="store_true", help="stop after the data gate")
    ap.add_argument("--admission", action="store_true", help="the bound section-3 admission gate")
    ap.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "d2_nba_cache")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck()
    elif a.admission:
        admission(a.cache)
    else:
        run(a.cache, a.gate)


if __name__ == "__main__":
    sys.exit(main())
