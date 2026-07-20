"""Round-robin tournament + ratings for arena checkpoints.

:func:`run_arena_eval` answers "does A beat B?" for one pairing. A tournament
runs that across every pair in a pool of checkpoints and condenses the results
into a single ranking. Ratings use the Bradley-Terry model (an order-independent
maximum-likelihood fit to the full win matrix) reported on the familiar Elo
scale, alongside the raw per-pairing win rates.

Draws (simultaneous KO) and timeouts count as half a win to each side for rating
purposes; they are still reported separately in the standings.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from rl_framework.training.arena_eval import run_arena_eval, run_n_agent_eval

# Elo scale constants: a 400-point gap ⇒ ~10:1 expected odds.
_ELO_BASE = 1500.0
_ELO_SCALE = 400.0


def _unique_labels(paths: list[str]) -> list[str]:
    """Return short, unique display labels for *paths*.

    Prefers the bare file stem; falls back to the last two path components, then
    to an indexed suffix, so checkpoints from different runs that share a name
    (e.g. several ``final_model.zip``) stay distinguishable.
    """
    stems = [Path(p).stem for p in paths]
    if len(set(stems)) == len(stems):
        return stems
    two = ["/".join(Path(p).with_suffix("").parts[-2:]) for p in paths]
    if len(set(two)) == len(two):
        return two
    out: list[str] = []
    seen: dict[str, int] = {}
    counts = Counter(stems)
    for s in stems:
        seen[s] = seen.get(s, 0) + 1
        out.append(s if counts[s] == 1 else f"{s}#{seen[s]}")
    return out


def resolve_competitors(
    checkpoints: list[str], include_random: bool = False
) -> list[tuple[str, str]]:
    """Expand *checkpoints* (files and/or directories) into ``(label, path)`` pairs.

    Directory arguments contribute every ``*.zip`` they contain (non-recursive,
    sorted). Duplicate paths are dropped, order preserved. When *include_random*
    is set, a ``("random", "random")`` baseline is appended.
    """
    expanded: list[str] = []
    for item in checkpoints:
        p = Path(item)
        if p.is_dir():
            expanded.extend(str(f) for f in sorted(p.glob("*.zip")))
        else:
            expanded.append(str(item))
    # Drop duplicates, preserve first-seen order.
    seen: set[str] = set()
    unique_paths = [p for p in expanded if not (p in seen or seen.add(p))]
    labels = _unique_labels(unique_paths) if unique_paths else []
    competitors = list(zip(labels, unique_paths))
    if include_random:
        competitors.append(("random", "random"))
    return competitors


def _bradley_terry_elo(
    names: list[str],
    wins: dict[str, dict[str, float]],
    games: dict[str, dict[str, int]],
    smoothing: float = 1.0,
    iters: int = 1000,
    tol: float = 1e-9,
) -> dict[str, float]:
    """Fit Bradley-Terry strengths to the win matrix and map them to Elo.

    Uses the standard minorization-maximization update (Hunter 2004). Each
    played pair is given ``smoothing`` virtual drawn games so an undefeated or
    winless competitor yields a finite rating instead of diverging.
    """
    strengths = {a: 1.0 for a in names}
    opponents = {a: [b for b in names if b != a and games[a][b] > 0] for a in names}
    # Smoothed win totals and pair counts.
    wins_s = {
        a: sum(wins[a][b] for b in opponents[a]) + 0.5 * smoothing * len(opponents[a])
        for a in names
    }
    n_games = {a: {b: games[a][b] + smoothing for b in opponents[a]} for a in names}

    for _ in range(iters):
        new = {}
        for a in names:
            denom = sum(
                n_games[a][b] / (strengths[a] + strengths[b]) for b in opponents[a]
            )
            new[a] = (wins_s[a] / denom) if denom > 0 else strengths[a]
        # Normalize to geometric mean 1 so the Elo zero-point is stable.
        log_mean = sum(math.log(v) for v in new.values()) / len(new)
        scale = math.exp(log_mean)
        new = {a: v / scale for a, v in new.items()}
        delta = max(abs(new[a] - strengths[a]) for a in names)
        strengths = new
        if delta < tol:
            break

    return {a: _ELO_BASE + _ELO_SCALE * math.log10(strengths[a]) for a in names}


def run_tournament(
    checkpoints: list[str],
    cfg: dict,
    n_episodes: int = 50,
    swap_roles: bool = True,
    include_random: bool = False,
    output_path: str | None = None,
    markdown_path: str | None = None,
) -> dict:
    """Run a round-robin tournament over *checkpoints* and rate the field.

    Parameters
    ----------
    checkpoints:
        Checkpoint paths and/or directories (each directory contributes its
        ``*.zip`` files). Either entry of any pairing may be ``"random"``.
    cfg:
        Experiment config providing ``environment`` and ``seed`` (same shape
        :func:`run_arena_eval` expects).
    n_episodes:
        Episodes per spawn orientation for each pairing (see
        :func:`run_arena_eval`; doubled when *swap_roles* is set).
    include_random:
        Append a random-action baseline competitor.
    output_path, markdown_path:
        Optional destinations for the JSON result and the markdown report.

    Returns a dict with ``competitors``, ``ratings`` (Elo), ``standings``
    (ranked), ``win_rate_matrix``, ``matches`` (per-pairing detail), and the run
    parameters.
    """
    if int(cfg.get("environment", {}).get("num_agents", 2)) > 2:
        return run_n_agent_tournament(
            checkpoints, cfg, n_episodes=n_episodes, include_random=include_random,
            output_path=output_path, markdown_path=markdown_path,
        )
    competitors = resolve_competitors(checkpoints, include_random)
    if len(competitors) < 2:
        raise ValueError(
            f"A tournament needs at least 2 competitors, got {len(competitors)}. "
            "Pass more checkpoints or set include_random."
        )
    names = [c[0] for c in competitors]
    paths = {name: path for name, path in competitors}

    wins = {a: {b: 0.0 for b in names} for a in names}
    games = {a: {b: 0 for b in names} for a in names}
    win_rate_matrix = {a: {b: None for b in names} for a in names}
    matches: list[dict[str, Any]] = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            res = run_arena_eval(
                paths[a], paths[b], cfg, n_episodes=n_episodes, swap_roles=swap_roles
            )
            total = res["n_episodes"]
            a_wins = res["policy_win_rate"] * total
            b_wins = res["opponent_win_rate"] * total
            # Draws + timeouts split evenly as half-wins for rating purposes.
            half = (res["draw_rate"] + res["timeout_rate"]) * total / 2.0
            wins[a][b] += a_wins + half
            wins[b][a] += b_wins + half
            games[a][b] += total
            games[b][a] += total
            win_rate_matrix[a][b] = res["policy_win_rate"]
            win_rate_matrix[b][a] = res["opponent_win_rate"]
            matches.append(
                {
                    "competitor": a,
                    "opponent": b,
                    "competitor_win_rate": res["policy_win_rate"],
                    "opponent_win_rate": res["opponent_win_rate"],
                    "draw_rate": res["draw_rate"],
                    "timeout_rate": res["timeout_rate"],
                    "competitor_episode_metrics": res.get(
                        "policy_episode_metrics", {}
                    ),
                    "opponent_episode_metrics": res.get(
                        "opponent_episode_metrics", {}
                    ),
                    "n_episodes": total,
                }
            )

    ratings = _bradley_terry_elo(names, wins, games)

    standings = []
    for name in names:
        total_games = sum(games[name][b] for b in names)
        # Aggregate W/L/D from the raw match records.
        agg = _aggregate_record(matches, name)
        standings.append(
            {
                "competitor": name,
                "elo": round(ratings[name], 1),
                "wins": agg["wins"],
                "losses": agg["losses"],
                "draws": agg["draws"],
                "games": total_games,
                "win_rate": (agg["wins"] / total_games) if total_games else 0.0,
            }
        )
    standings.sort(key=lambda s: s["elo"], reverse=True)
    for rank, s in enumerate(standings, start=1):
        s["rank"] = rank

    result = {
        "competitors": [{"label": n, "path": paths[n]} for n in names],
        "ratings": {n: round(ratings[n], 1) for n in names},
        "standings": standings,
        "win_rate_matrix": win_rate_matrix,
        "matches": matches,
        "n_episodes": n_episodes,
        "swap_roles": swap_roles,
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if markdown_path is not None:
        md = Path(markdown_path)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(format_markdown(result), encoding="utf-8")
    return result


def run_n_agent_tournament(
    checkpoints: list[str], cfg: dict, n_episodes: int = 50,
    include_random: bool = False, output_path: str | None = None,
    markdown_path: str | None = None,
) -> dict:
    """Rate an N-agent field by rotating competitors through all arena slots."""
    competitors = resolve_competitors(checkpoints, include_random)
    slots = int(cfg["environment"].get("num_agents", 2))
    if len(competitors) < slots:
        raise ValueError(f"An N-agent tournament needs at least {slots} competitors")
    names = [label for label, _ in competitors]
    paths = dict(competitors)
    wins = {a: {b: 0.0 for b in names} for a in names}
    games = {a: {b: 0 for b in names} for a in names}
    records = {name: {"wins": 0, "losses": 0, "draws": 0, "games": 0} for name in names}
    matches = []
    # Sliding groups keep cost bounded while ensuring every competitor appears.
    for start in range(len(names)):
        group = [names[(start + offset) % len(names)] for offset in range(slots)]
        result = run_n_agent_eval([paths[name] for name in group], cfg, n_episodes=n_episodes)
        scores = result["agent_mean_scores"]
        for slot, name in enumerate(group):
            agent = f"agent_{slot}"
            records[name]["games"] += n_episodes
            records[name]["wins"] += round(result["agent_win_rates"][agent] * n_episodes)
            records[name]["draws"] += round((result["draw_rate"] + result["timeout_rate"]) * n_episodes)
        for i, a in enumerate(group):
            for j, b in enumerate(group):
                if i >= j:
                    continue
                a_score, b_score = scores[f"agent_{i}"], scores[f"agent_{j}"]
                wins[a][b] += a_score * n_episodes
                wins[b][a] += b_score * n_episodes
                games[a][b] += n_episodes
                games[b][a] += n_episodes
        matches.append({"competitors": group, **result})
    ratings = _bradley_terry_elo(names, wins, games)
    standings = []
    for name in names:
        record = records[name]
        record["losses"] = max(record["games"] - record["wins"] - record["draws"], 0)
        standings.append({"competitor": name, "elo": round(ratings[name], 1), **record,
                          "win_rate": record["wins"] / max(record["games"], 1)})
    standings.sort(key=lambda item: item["elo"], reverse=True)
    for rank, item in enumerate(standings, 1):
        item["rank"] = rank
    output = {"competitors": [{"label": n, "path": paths[n]} for n in names], "ratings": {n: round(ratings[n], 1) for n in names}, "standings": standings, "matches": matches, "n_episodes": n_episodes, "n_agents": slots, "swap_roles": "slot_rotation"}
    if output_path:
        Path(output_path).write_text(json.dumps(output, indent=2), encoding="utf-8")
    if markdown_path:
        Path(markdown_path).write_text(format_markdown(output), encoding="utf-8")
    return output


def _aggregate_record(matches: list[dict[str, Any]], name: str) -> dict[str, int]:
    """Aggregate integer W/L/D counts for *name* across all its matches.

    Draws and timeouts both count toward draws (neither side scored a KO).
    """
    wins = losses = draws = 0
    for m in matches:
        if name not in (m["competitor"], m["opponent"]):
            continue
        total = m["n_episodes"]
        if m["competitor"] == name:
            my_wr, opp_wr = m["competitor_win_rate"], m["opponent_win_rate"]
        else:
            my_wr, opp_wr = m["opponent_win_rate"], m["competitor_win_rate"]
        wins += round(my_wr * total)
        losses += round(opp_wr * total)
        draws += round((m["draw_rate"] + m["timeout_rate"]) * total)
    return {"wins": wins, "losses": losses, "draws": draws}


def format_markdown(result: dict) -> str:
    """Render a tournament *result* dict as a markdown report."""
    lines: list[str] = ["# Arena Tournament Results", ""]
    lines.append(
        f"{len(result['competitors'])} competitors · "
        f"{result['n_episodes']} episodes/orientation · "
        f"swap_roles={result['swap_roles']}"
    )
    lines.append("")
    lines.append("## Standings")
    lines.append("")
    lines.append("| Rank | Competitor | Elo | W | L | D | Games | Win% |")
    lines.append("|-----:|------------|----:|--:|--:|--:|------:|-----:|")
    for s in result["standings"]:
        lines.append(
            f"| {s['rank']} | {s['competitor']} | {s['elo']:.0f} | "
            f"{s['wins']} | {s['losses']} | {s['draws']} | {s['games']} | "
            f"{s['win_rate'] * 100:.1f}% |"
        )
    lines.append("")
    if "win_rate_matrix" not in result:
        return "\n".join(lines)
    lines.append("## Win-rate matrix (row vs column)")
    lines.append("")
    names = [c["label"] for c in result["competitors"]]
    header = "| vs | " + " | ".join(names) + " |"
    sep = "|----|" + "|".join("----:" for _ in names) + "|"
    lines.append(header)
    lines.append(sep)
    matrix = result["win_rate_matrix"]
    for a in names:
        cells = []
        for b in names:
            v = matrix[a][b]
            cells.append("—" if a == b or v is None else f"{v * 100:.0f}%")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)
