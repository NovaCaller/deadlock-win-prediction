from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


class Player(BaseModel):
    model_config = {'validate_assignment': True}

    hero_name: str
    net_worth: Annotated[int, Field(ge=0)]
    ability_points: Annotated[int, Field(ge=0)]
    level: Annotated[int, Field(ge=0)]

    _allowed_heroes: set[str] = set()

    @classmethod
    def set_allowed_heroes(cls, heroes: list[str]):
        cls._allowed_heroes = set(heroes)

    @field_validator("hero_name")
    @classmethod
    def validate_hero(cls, v: str) -> str:
        if v not in cls._allowed_heroes:
            raise ValueError(f"Unknown hero_name '{v}'")
        return v


class Objective(str, Enum):
    CORE = "Core"
    TIER_1_LANE_1 = "Tier1Lane1"
    TIER_1_LANE_3 = "Tier1Lane3"
    TIER_1_LANE_4 = "Tier1Lane4"
    TIER_2_LANE_1 = "Tier2Lane1"
    TIER_2_LANE_3 = "Tier2Lane3"
    TIER_2_LANE_4 = "Tier2Lane4"
    BARRACK_BOSS_LANE_1 = "BarrackBossLane1"
    BARRACK_BOSS_LANE_3 = "BarrackBossLane3"
    BARRACK_BOSS_LANE_4 = "BarrackBossLane4"
    TITAN = "Titan"
    TITAN_SHIELD_GENERATOR_1 = "TitanShieldGenerator1"
    TITAN_SHIELD_GENERATOR_2 = "TitanShieldGenerator2"


class Team(BaseModel):
    model_config = {'validate_assignment': True}

    players: Annotated[list[Player], Field(min_length=6, max_length=6)]
    lost_objectives: dict[Objective, bool] = Field(
        default_factory=lambda: {obj: False for obj in Objective}
    )

    @field_validator("lost_objectives")
    @classmethod
    def enforce_enum_order(cls, v):
        return {obj: v.get(obj, False) for obj in Objective}

    def update_player(self, player_idx: int, **updates) -> 'Team':
        new_players = self.players.copy()
        new_players[player_idx] = new_players[player_idx].model_copy(update=updates)
        return self.model_copy(update={'players': new_players})

    def update_objective(self, objective: Objective, is_destroyed: bool) -> 'Team':
        new_objectives = self.lost_objectives.copy()
        new_objectives[objective] = is_destroyed
        return self.model_copy(update={'lost_objectives': new_objectives})


class GameState(BaseModel):
    timestamp: Annotated[int, Field(ge=0)]
    team0: Team
    team1: Team

    def update_team0_player(self, player_idx: int, **updates) -> 'GameState':
        return self.model_copy(update={
            'team0': self.team0.update_player(player_idx, **updates)
        })

    def update_team1_player(self, player_idx: int, **updates) -> 'GameState':
        return self.model_copy(update={
            'team1': self.team1.update_player(player_idx, **updates)
        })

    def update_team0_objective(self, objective: Objective, is_destroyed: bool) -> 'GameState':
        return self.model_copy(update={
            'team0': self.team0.update_objective(objective, is_destroyed)
        })

    def update_team1_objective(self, objective: Objective, is_destroyed: bool) -> 'GameState':
        return self.model_copy(update={
            'team1': self.team1.update_objective(objective, is_destroyed)
        })


    