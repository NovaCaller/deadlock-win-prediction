from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator

class Hero(str, Enum):
    INFERNUS = "Infernus"
    SEVEN = "Seven"
    VINDICTA = "Vindicta"
    LADY_GEIST = "Lady Geist"
    ABRAMS = "Abrams"
    WRAITH = "Wraith"
    MCGINNIS = "McGinnis"
    PARADOX = "Paradox"
    DYNAMO = "Dynamo"
    KELVIN = "Kelvin"
    HAZE = "Haze"
    HOLLIDAY = "Holliday"
    BEBOP = "Bebop"
    CALICO = "Calico"
    GREY_TALON = "Grey Talon"
    MO_AND_KRILL = "Mo & Krill"
    SHIV = "Shiv"
    IVY = "Ivy"
    WARDEN = "Warden"
    YAMATO = "Yamato"
    LASH = "Lash"
    VISCOUS = "Viscous"
    POCKET = "Pocket"
    MIRAGE = "Mirage"
    VYPER = "Vyper"
    SINCLAIR = "Sinclair"
    MINA = "Mina"
    DRIFTER = "Drifter"
    VICTOR = "Victor"
    PAIGE = "Paige"
    THE_DOORMAN = "The Doorman"
    BILLY = "Billy"


class Player(BaseModel):
    model_config = {'validate_assignment': True}

    hero: Hero
    net_worth: Annotated[int, Field(ge=0)]
    ability_points: Annotated[int, Field(ge=0)]
    level: Annotated[int, Field(ge=0)]


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


    