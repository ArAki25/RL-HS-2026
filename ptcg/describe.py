"""Human-readable descriptions of observations and options (for the play UI)."""

from __future__ import annotations

import re

from .enums import AreaType, CardType, OptionType, SelectContext
from .observation import resolve_card, resolve_energy, resolve_target

AREA_NAMES = {
    AreaType.DECK: "Deck",
    AreaType.HAND: "Hand",
    AreaType.DISCARD: "Ablage",
    AreaType.ACTIVE: "Aktiv",
    AreaType.BENCH: "Bank",
    AreaType.PRIZE: "Preiskarten",
    AreaType.STADIUM: "Stadion",
    AreaType.LOOKING: "aufgedeckt",
}

CARD_TYPE_NAMES = {
    CardType.POKEMON: "Pokémon",
    CardType.ITEM: "Item",
    CardType.TOOL: "Ausrüstung",
    CardType.SUPPORTER: "Unterstützer",
    CardType.STADIUM: "Stadion",
    CardType.BASIC_ENERGY: "Basis-Energie",
    CardType.SPECIAL_ENERGY: "Spezial-Energie",
}

CONTEXT_PROMPTS = {
    SelectContext.MAIN: "Dein Zug: wähle eine Aktion",
    SelectContext.SETUP_ACTIVE_POKEMON: "Wähle dein aktives Pokémon",
    SelectContext.SETUP_BENCH_POKEMON: "Wähle Pokémon für die Bank",
    SelectContext.SWITCH: "Wähle ein Pokémon zum Tauschen",
    SelectContext.TO_ACTIVE: "Wähle ein neues aktives Pokémon",
    SelectContext.TO_BENCH: "Wähle Karten für die Bank",
    SelectContext.TO_HAND: "Wähle Karten für deine Hand",
    SelectContext.DISCARD: "Wähle Karten zum Ablegen",
    SelectContext.TO_DECK: "Wähle Karten, die ins Deck zurückgehen",
    SelectContext.TO_DECK_BOTTOM: "Wähle Karten für unter das Deck",
    SelectContext.DAMAGE_COUNTER: "Wähle, wohin Schadensmarken kommen",
    SelectContext.HEAL: "Wähle ein Pokémon zum Heilen",
    SelectContext.ATTACH_FROM: "Wähle die Karte zum Anlegen",
    SelectContext.ATTACH_TO: "Wähle, woran angelegt wird",
    SelectContext.DISCARD_ENERGY: "Wähle Energie zum Ablegen",
    SelectContext.SKILL_ORDER: "Wähle die Reihenfolge der Effekte",
    SelectContext.ATTACK: "Wähle einen Angriff",
    SelectContext.EVOLVE: "Wähle eine Entwicklung",
    SelectContext.DRAW_COUNT: "Wie viele Karten ziehen?",
    SelectContext.IS_FIRST: "Möchtest du beginnen?",
    SelectContext.MULLIGAN: "Mulligan: zusätzliche Karte ziehen?",
    SelectContext.ACTIVATE: "Effekt aktivieren?",
}


class Describer:
    def __init__(self, cards: dict[int, dict], attacks: dict[int, dict]):
        self.cards = cards
        self.attacks = attacks
        # Energy type -> symbol, read from basic energy names like "Basic {W} Energy".
        self.energy_symbols = {0: "C"}
        for c in cards.values():
            match = re.search(r"\{(\w)\}", c["name"])
            if c["cardType"] == CardType.BASIC_ENERGY and match:
                self.energy_symbols[c["energyType"]] = match.group(1)

    def name(self, card_id: int | None) -> str:
        c = self.cards.get(card_id)
        return c["name"] if c else "?"

    def energies(self, types: list[int]) -> str:
        return "".join(self.energy_symbols.get(t, "?") for t in types)

    def card_text(self, card_id: int) -> str:
        """Tooltip text: type, HP, attacks and card text."""
        c = self.cards.get(card_id)
        if c is None:
            return ""
        lines = [CARD_TYPE_NAMES.get(c["cardType"], "")]
        if c["cardType"] == CardType.POKEMON:
            stage = "Basis" if c["basic"] else f"entwickelt sich aus {c['evolvesFrom']}"
            lines[0] += f" · {c['hp']} KP · {stage} · Rückzug {c['retreatCost']}"
        for skill in c["skills"]:
            lines.append(f"{skill['name']}: {skill['text']}")
        for attack_id in c["attacks"]:
            a = self.attacks[attack_id]
            lines.append(f"[{self.energies(a['energies'])}] {a['name']} {a['damage'] or ''} {a['text']}".strip())
        return "\n".join(lines)

    def card(self, card: dict | None) -> dict:
        if card is None:
            return {"name": "verdeckt", "text": "", "id": None}
        return {"name": self.name(card["id"]), "text": self.card_text(card["id"]), "id": card["id"]}

    def pokemon(self, p: dict | None) -> dict:
        if p is None:  # face down during setup
            return {**self.card(None), "hp": 0, "maxHp": 0, "energies": "", "tools": []}
        return {
            **self.card(p),
            "hp": p["hp"],
            "maxHp": p["maxHp"],
            "energies": self.energies(p["energies"]),
            "tools": [self.name(t["id"]) for t in p["tools"]],
        }

    def player(self, player: dict) -> dict:
        return {
            "active": [self.pokemon(p) for p in player["active"]],
            "bench": [self.pokemon(p) for p in player["bench"]],
            "benchMax": player.get("benchMax", 5),
            "hand": [self.card(c) for c in player["hand"]] if player.get("hand") is not None else None,
            "handCount": player["handCount"],
            "deckCount": player["deckCount"],
            "discard": [self.card(c) for c in player["discard"]],
            "prizes": len(player["prize"]),
            "status": [s for s in ("poisoned", "burned", "asleep", "paralyzed", "confused") if player.get(s)],
        }

    def board(self, obs: dict, seat: int | None = None) -> dict:
        """The board from ``seat``'s point of view (default: the player to move)."""
        current = obs["current"]
        me = current["yourIndex"] if seat is None else seat
        stadium = (current.get("stadium") or [None])[0]
        return {
            "turn": current["turn"],
            "you": self.player(current["players"][me]),
            "opponent": self.player(current["players"][1 - me]),
            "stadium": self.card(stadium) if stadium else None,
        }

    def prompt(self, obs: dict) -> str:
        select = obs["select"]
        context = select["context"]
        text = CONTEXT_PROMPTS.get(context)
        if text is None:
            try:
                text = SelectContext(context).name.replace("_", " ").capitalize()
            except ValueError:
                text = f"Auswahl ({context})"
        source = select.get("effect") or select.get("contextCard")
        if source:
            text += f" — {self.name(source['id'])}"
        if select.get("remainEnergyCost"):
            text += f" (noch {select['remainEnergyCost']} Energie)"
        if select.get("remainDamageCounter"):
            text += f" (noch {select['remainDamageCounter']} Schadensmarken)"
        return text

    def _where(self, option: dict, obs: dict) -> str:
        area = AREA_NAMES.get(option.get("area"), "")
        if option.get("playerIndex") not in (None, obs["current"]["yourIndex"]):
            area += " (Gegner)"
        return area

    def option(self, option: dict, obs: dict) -> str:
        kind = option["type"]
        me = obs["current"]["players"][obs["current"]["yourIndex"]]
        hand = me.get("hand") or []

        def hand_card() -> str:
            i = option.get("index")
            return self.name(hand[i]["id"]) if i is not None and i < len(hand) else "?"

        def target() -> str:
            t = resolve_target(option, obs)
            return self.name(t["id"]) if t else "?"

        if kind == OptionType.NUMBER:
            return str(option["number"])
        if kind == OptionType.YES:
            return "Ja"
        if kind == OptionType.NO:
            return "Nein"
        if kind in (OptionType.CARD, OptionType.TOOL_CARD, OptionType.ENERGY_CARD):
            card = resolve_card(option, obs)
            if card is None:
                return f"Verdeckte Karte ({self._where(option, obs)})"
            label = self.name(card["id"])
            if "hp" in card:
                label += f" {card['hp']}/{card['maxHp']} KP"
            return f"{label} · {self._where(option, obs)}"
        if kind == OptionType.ENERGY:
            energy = resolve_energy(option, obs)
            pokemon = resolve_card(option, obs)
            return f"{self.name(energy['id']) if energy else 'Energie'} von {self.name(pokemon['id']) if pokemon else '?'}"
        if kind == OptionType.PLAY:
            return f"Spielen: {hand_card()}"
        if kind == OptionType.ATTACH:
            return f"Anlegen: {hand_card()} → {target()} ({AREA_NAMES.get(option['inPlayArea'], '')})"
        if kind == OptionType.EVOLVE:
            return f"Entwickeln: {target()} → {hand_card()}"
        if kind == OptionType.ABILITY:
            pokemon = resolve_card(option, obs)
            card = self.cards.get(pokemon["id"]) if pokemon else None
            skill = card["skills"][0]["name"] if card and card["skills"] else "Fähigkeit"
            return f"Fähigkeit: {skill} ({self.name(pokemon['id']) if pokemon else '?'})"
        if kind == OptionType.DISCARD:
            return "Abwerfen"
        if kind == OptionType.RETREAT:
            return "Rückzug"
        if kind == OptionType.ATTACK:
            a = self.attacks.get(option["attackId"])
            if a is None:
                return "Angriff"
            return f"Angriff: {a['name']} [{self.energies(a['energies'])}] {a['damage'] or ''}".strip()
        if kind == OptionType.END:
            return "Zug beenden"
        if kind == OptionType.SKILL:
            return f"Effekt: {self.name(option.get('cardId'))}"
        if kind == OptionType.SPECIAL_CONDITION:
            return "Spezialzustand"
        return f"Option {kind}"

    def option_text(self, option: dict, obs: dict) -> str:
        """Tooltip for an option: the text of the card it refers to."""
        kind = option["type"]
        card = None
        if kind in (OptionType.PLAY, OptionType.ATTACH, OptionType.EVOLVE):
            hand = obs["current"]["players"][obs["current"]["yourIndex"]].get("hand") or []
            i = option.get("index")
            card = hand[i] if i is not None and i < len(hand) else None
        elif kind == OptionType.ATTACK:
            a = self.attacks.get(option["attackId"])
            return a["text"] if a else ""
        elif kind == OptionType.SKILL:
            return self.card_text(option.get("cardId"))
        else:
            card = resolve_card(option, obs)
        return self.card_text(card["id"]) if card else ""
