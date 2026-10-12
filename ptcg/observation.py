"""Helpers for reading engine observations. Engine-free, so it is bundled into the submission."""

from __future__ import annotations

from .enums import AreaType


def resolve_card(option: dict, obs: dict) -> dict | None:
    """Return the card (or in-play Pokémon) an option refers to, if it can be found."""
    area, index = option.get("area"), option.get("index")
    if area is None or index is None:
        return None
    current = obs["current"]
    player = current["players"][option.get("playerIndex", current["yourIndex"])]
    zones = {
        AreaType.DECK: obs["select"].get("deck"),
        AreaType.HAND: player["hand"],
        AreaType.DISCARD: player["discard"],
        AreaType.ACTIVE: player["active"],
        AreaType.BENCH: player["bench"],
        AreaType.PRIZE: player["prize"],
        AreaType.STADIUM: current.get("stadium"),
        AreaType.LOOKING: current.get("looking"),
    }
    zone = zones.get(area)
    if not zone or index >= len(zone):
        return None
    return zone[index]


def resolve_target(option: dict, obs: dict) -> dict | None:
    """Return our in-play Pokémon that an ATTACH/EVOLVE option targets."""
    area, index = option.get("inPlayArea"), option.get("inPlayIndex")
    if area is None or index is None:
        return None
    me = obs["current"]["players"][obs["current"]["yourIndex"]]
    zone = me["active"] if area == AreaType.ACTIVE else me["bench"] if area == AreaType.BENCH else None
    if not zone or index >= len(zone):
        return None
    return zone[index]


def resolve_energy(option: dict, obs: dict) -> dict | None:
    """Return the attached energy card an ENERGY option refers to."""
    pokemon = resolve_card(option, obs)
    index = option.get("energyIndex")
    if pokemon is None or index is None:
        return None
    cards = pokemon.get("energyCards") or []
    return cards[index] if index < len(cards) else None
