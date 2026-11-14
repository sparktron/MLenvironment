"""Wizard schema for this task."""


def gui_schema(_training_schema, _eval_schema):
    schema = {
            "environment": {
                "type": {"value": "organism_arena_parallel", "type": "fixed"},
                "seed": {"value": 11, "type": "int", "desc": "Environment random seed"},
                "num_agents": {
                    "value": 2,
                    "type": "int",
                    "desc": "Number of organisms (2 = duel, >2 = free-for-all)",
                    "min": 2,
                    "max": 8,
                },
                "sim": {
                    "arena_half_extent": {
                        "value": 1.2,
                        "type": "float",
                        "desc": "Arena half-size",
                        "min": 0.5,
                        "max": 5,
                    },
                    "move_speed": {
                        "value": 0.05,
                        "type": "float",
                        "desc": "Max movement per step (arena units)",
                        "min": 0.01,
                        "max": 0.5,
                    },
                    "spawn_jitter": {
                        "value": 0.1,
                        "type": "float",
                        "desc": "Spawn position jitter half-width (0 = fixed spawns)",
                        "min": 0,
                        "max": 1,
                    },
                    "collision_radius": {
                        "value": 0.08,
                        "type": "float",
                        "desc": "Per-unit-size body radius; min separation = radius x (sizes summed)",
                        "min": 0.01,
                        "max": 0.5,
                    },
                    "speed_size_exponent": {
                        "value": 1.0,
                        "type": "float",
                        "desc": "How strongly size slows movement (0 = size-independent speed)",
                        "min": 0,
                        "max": 3,
                    },
                },
                "morphology": {
                    "base_size": {
                        "value": 1.0,
                        "type": "float",
                        "desc": "Base organism size",
                        "min": 0.1,
                        "max": 5,
                    },
                    "episode_growth_scale": {
                        "value": 0.0,
                        "type": "float",
                        "desc": "Size growth per step (scales damage and max health)",
                        "min": 0,
                        "max": 0.1,
                    },
                    "health": {
                        "value": 1.2,
                        "type": "float",
                        "desc": "Health pool (scaled by size each step)",
                        "min": 0.1,
                        "max": 10,
                    },
                },
                "battle_rules": {
                    "damage": {
                        "value": 0.06,
                        "type": "float",
                        "desc": "Damage per hit",
                        "min": 0.01,
                        "max": 1,
                    },
                    "collision_damage": {
                        "value": 0.0,
                        "type": "float",
                        "desc": "Optional contact damage scaled by the other organism's size",
                        "min": 0,
                        "max": 1,
                    },
                    "attack_range": {
                        "value": 0.2,
                        "type": "float",
                        "desc": "Attack range",
                        "min": 0.05,
                        "max": 2,
                    },
                    "cooldown_steps": {
                        "value": 3,
                        "type": "int",
                        "desc": "Cooldown between attacks",
                        "min": 0,
                        "max": 20,
                    },
                    "sensing_radius": {
                        "value": 2.0,
                        "type": "float",
                        "desc": "Distance at which an opponent becomes visible",
                        "min": 0.1,
                        "max": 10,
                    },
                    "attack_falloff": {
                        "value": "linear",
                        "type": "choice",
                        "choices": ["linear", "binary"],
                        "desc": "Damage falloff with distance: linear (graded) or binary (hard cliff)",
                    },
                    "max_steps": {
                        "value": 400,
                        "type": "int",
                        "desc": "Max steps per episode",
                        "min": 50,
                        "max": 10000,
                    },
                    "win_health_threshold": {
                        "value": 0.0,
                        "type": "float",
                        "desc": "Health below which opponent loses",
                        "min": 0,
                        "max": 1,
                    },
                },
                "resources": {
                    "initial_energy": {
                        "value": 1.0,
                        "type": "float",
                        "desc": "Energy at spawn (must be <= max energy)",
                        "min": 0.01,
                        "max": 10,
                    },
                    "max_energy": {
                        "value": 1.0,
                        "type": "float",
                        "desc": "Energy cap; food refills up to this",
                        "min": 0.01,
                        "max": 10,
                    },
                    "movement_cost": {
                        "value": 0.01,
                        "type": "float",
                        "desc": "Energy drained per unit of movement",
                        "min": 0.0001,
                        "max": 1,
                    },
                    "attack_cost": {
                        "value": 0.04,
                        "type": "float",
                        "desc": "Energy spent per attack (attack fails if unaffordable)",
                        "min": 0.0001,
                        "max": 1,
                    },
                    "food_count": {
                        "value": 2,
                        "type": "int",
                        "desc": "Simultaneous food pellets in the arena",
                        "min": 0,
                        "max": 20,
                    },
                    "food_energy": {
                        "value": 0.35,
                        "type": "float",
                        "desc": "Energy restored per pellet eaten",
                        "min": 0.01,
                        "max": 10,
                    },
                    "food_radius": {
                        "value": 0.10,
                        "type": "float",
                        "desc": "Pickup distance for a food pellet",
                        "min": 0.01,
                        "max": 1,
                    },
                    "food_respawn_steps": {
                        "value": 40,
                        "type": "int",
                        "desc": "Steps before an eaten pellet reappears",
                        "min": 0,
                        "max": 2000,
                    },
                    "food_placement": {
                        "value": "uniform",
                        "type": "choice",
                        "choices": ["uniform", "center"],
                        "desc": "Uniform food or a contested central food patch",
                    },
                },
                "reward": {
                    "approach_weight": {
                        "value": 0.0,
                        "type": "float",
                        "desc": "Reward weight for bounded progress toward the nearest opponent",
                        "min": 0,
                        "max": 1,
                    },
                },
            },
            "training": _training_schema(),
            "evaluation": _eval_schema(),
            "extra": _arena_extra_schema(),
        }
    schema["training"]["num_envs"].update(
        {
            "value": 1,
            "desc": (
                "Parallel environments (shared-policy arena requires 1; "
                "self-play configs can use more)"
            ),
            "max": 32,
        }
    )
    return schema


def _arena_extra_schema() -> dict:
    """Top-level (not ``environment``-nested) groups for the self-play arena
    path: league self-play, dense-to-sparse reward annealing, and the
    win-rate-gated curriculum. These are the arena's documented preferred
    training path (see ``docs/organism_arena_config.md``) but previously had
    no wizard fields at all, so building one from scratch required hand-editing
    YAML — only loading the bundled ``organisms_fight_arena`` template exposed
    them. Defaults mirror that bundled config."""
    return {
        "self_play": {
            "enabled": {
                "value": False,
                "type": "bool",
                "desc": "Train agent_0 against frozen past-self snapshots (agent_1)",
            },
            "snapshot_freq": {
                "value": 5000,
                "type": "int",
                "desc": "Save a league snapshot every N env timesteps",
                "min": 100,
                "max": 1_000_000,
            },
            "max_league_size": {
                "value": 10,
                "type": "int",
                "desc": "Max snapshots retained (oldest pruned first)",
                "min": 1,
                "max": 50,
            },
            "sampling_mode": {
                "value": "uniform",
                "type": "choice",
                "choices": ["uniform", "recent_bias"],
                "desc": "How opponents are sampled from the league",
            },
        },
        "reward_annealing": {
            "enabled": {
                "value": False,
                "type": "bool",
                "desc": "Ramp the dense per-hit reward to 0 so the terminal win/loss signal dominates",
            },
            "anneal_steps": {
                "value": 15000,
                "type": "int",
                "desc": "Timesteps over which damage reward ramps down after the elimination gate opens",
                "min": 100,
                "max": 10_000_000,
            },
            "min_eliminations": {
                "value": 10,
                "type": "int",
                "desc": "Observed non-timeout outcomes required before damage reward annealing starts",
                "min": 0,
                "max": 1_000_000,
            },
        },
        "curriculum": {
            "enabled": {
                "value": False,
                "type": "bool",
                "desc": "Ramp battle_rules difficulty as the configured metric improves",
            },
            "metric": {
                "value": "arena/agent_0_win_rate",
                "type": "text",
                "desc": "Logger/ep_info_buffer key gating level-ups",
            },
            "warmup_steps": {
                "value": 5000,
                "type": "int",
                "desc": "Hold level-ups until this many timesteps have elapsed",
                "min": 0,
                "max": 10_000_000,
            },
            "max_level": {
                "value": 3,
                "type": "int",
                "desc": "Highest curriculum level reachable",
                "min": 1,
                "max": 10,
            },
            "level_up_thresholds": {
                "value": {"0": 0.6, "1": 0.65, "2": 0.7},
                "type": "json",
                "desc": 'Per-level metric threshold to advance, e.g. {"0": 0.6, "1": 0.65}',
            },
            "level_params": {
                "value": {
                    "1": {
                        "battle_rules.cooldown_steps": 3,
                        "battle_rules.damage": 0.08,
                    },
                    "2": {
                        "battle_rules.cooldown_steps": 2,
                        "battle_rules.damage": 0.10,
                    },
                    "3": {
                        "battle_rules.cooldown_steps": 2,
                        "battle_rules.damage": 0.12,
                    },
                },
                "type": "json",
                "desc": "Per-level config overrides applied on level-up (dotted-key: value)",
            },
        },
    }
