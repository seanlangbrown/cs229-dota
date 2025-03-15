import boto3
import pandas as pd
import numpy as np
import json
from functools import partial
import math
from sklearn.model_selection import train_test_split
from cs229_utils import print_full_df, print_full_columns
import torch
from torch.utils.data import Dataset, DataLoader
from normalize import load_range_tracker
import sys
import traceback
import os


pd.set_option('future.no_silent_downcasting', True)

def get_stacktrace():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    
    return traceback.format_exception(exc_type, exc_value, exc_traceback)
    

# our data dictionary
data_dictionary_fields = {
    "time": {
        "dfname": "time",
        "cumulative": False,
        "type": "game",
        "description": "time since the start of the game, in seconds",
    },
    "gold": {
        "dfname": "gold",
        "cumulative": False,
        "type": "player",
        "description": "gold available for purchases, earned while playing the game",
    },
    "lh": {
        "dfname": "lh",
        "cumulative": False,
        "type": "player",
        "description": "continuous measurement of health",
    },
    "xp": {
        "dfname": "xp",
        "cumulative": False,
        "type": "player",
        "description": "experience points gained by fighting in the game",
    },
    "upgrade_ability_1": {
        "dfname": "upgrade_ability_1",
        "cumulative": False,
        "type": "player",
        "description": "ability that was upgraded, will be NaN at timestamps with no upgrade. There are 2095 abilities, each represented by a four digit code.",
    },
    "upgrade_ability_2": {
        "dfname": "upgrade_ability_2",
        "cumulative": False,
        "type": "player",
        "description": "ability that was upgraded, will be NaN at timestamps with no upgrade",
    },
    "upgrade_ability_3": {
        "dfname": "upgrade_ability_3",
        "cumulative": False,
        "type": "player",
        "description": "ability that was upgraded, will be NaN at timestamps with no upgrade",
    },
    "upgrade_ability_4": {
        "dfname": "upgrade_ability_4",
        "cumulative": False,
        "type": "player",
        "description": "ability that was upgraded, will be NaN at timestamps with no upgrade",
    },
    "upgrade_ability_5": {
        "dfname": "upgrade_ability_5",
        "cumulative": False,
        "type": "player",
        "description": "ability that was upgraded, will be NaN at timestamps with no upgrade",
    },
    "upgrade_ability_level_1": {
        "dfname": "upgrade_ability_level_1",
        "cumulative": False,
        "type": "player",
        "description": "new level of ability that was upgraded",
    },
    "upgrade_ability_level_2": {
        "dfname": "upgrade_ability_level_2",
        "cumulative": False,
        "type": "player",
        "description": "new level of ability that was upgraded",
    },
    "upgrade_ability_level_3": {
        "dfname": "upgrade_ability_level_3",
        "cumulative": False,
        "type": "player",
        "description": "new level of ability that was upgraded",
    },
    "upgrade_ability_level_4": {
        "dfname": "upgrade_ability_level_4",
        "cumulative": False,
        "type": "player",
        "description": "new level of ability that was upgraded",
    },
    "upgrade_ability_level_5": {
        "dfname": "upgrade_ability_level_5",
        "cumulative": False,
        "type": "player",
        "description": "new level of ability that was upgraded",
    },
    "game_mode": {
        "dfname": "game_mode",
        "cumulative": False,
        "type": "player",
        "description": "game mode, see https://github.com/odota/dotaconstants/blob/master/build/game_mode.json",
    },
#     "active_team_fight": {
#     "dfname": "",
#     "cumulative": False,
#    "type": "game",
#     "description": "is there currently any active team fights. Team fights are identified by the game when players are close together and dealing damage.",
#     },
#     "team_fight_deaths_per_s": {
#     "dfname": "",
#     "cumulative": False,
#     "type": "teamfight",
#     "description": "a measurement of the intensity of the team fight. This statistic is 0 until the first death in a team fight and returns to zero after the fight ends",
#     },
#     "team_fight_deaths_smoothed": {
#     "dfname": "",
#     "cumulative": False,
#    "type": "teamfight",
#     "description": "The raw data does not include the time of deaths, so this attempts to approximate similar data by assuming deaths occur at a constant rate. This static linearly increases during a team fight from the first death until the last death",
#     },
#     "team_fight_damage_smoothed": {
#     "dfname": "",
#     "cumulative": False,
#    "type": "teamfight",
#     "description": "The raw data does not include damage with timestamps, so this approximates it by linearly smoothing the total damage in the fight over time",
#     },
#     "team_fight_damage_per_s": {
#     "dfname": "",
#     "cumulative": False,
#    "type": "teamfight",
#     "description": "this is a measure of the intensity of a team fight",
#     },
#     "in_range_team_fight": {   
#     "dfnames": [
#         "radiant_t1_top_in_range_team_fight",
#         "radiant_t1_mid_in_range_team_fight",
#         "radiant_t1_bottom_in_range_team_fight",
#         "radiant_t2_top_in_range_team_fight",
#         "radiant_t2_mid_in_range_team_fight",
#         "radiant_t2_bottom_in_range_team_fight",
#         "radiant_t3_top_in_range_team_fight",
#         "radiant_t3_mid_in_range_team_fight",
#         "radiant_t3_bottom_in_range_team_fight",
#         "radiant_t4_base_in_range_team_fight",
#         "dire_t1_top_in_range_team_fight",
#         "dire_t1_mid_in_range_team_fight",
#         "dire_t1_bottom_in_range_team_fight",
#         "dire_t2_top_in_range_team_fight",
#         "dire_t2_mid_in_range_team_fight",
#         "dire_t2_bottom_in_range_team_fight",
#         "dire_t3_top_in_range_team_fight",
#         "dire_t3_mid_in_range_team_fight",
#         "dire_t3_bottom_in_range_team_fight",
#         "dire_t4_base_in_range_team_fight",
#     ],
#     "cumulative": False,
#    "type": "teamfight",
#     "description": "These columns indicate if any team fight is with 844 units of a tower. Towers belong to team radiant or dire and have a t and bottom, mid, top position. 844 unites is the width of the tower, the maximum range of hero damage, plus a buffer. The location of the team fight is determined by finding the centroid of any deaths in the fight, since other locations are not provided in the raw data.",
#     },
#     "human_players": {
#     "dfname": "",
#     "cumulative": False,
#    "type": "game",
#     "description": "count of human players in the game. range 1 to 10",
#     },
#     "first_blood_time": {
#     "dfname": "",
#     "cumulative": False,
#    "type": "game",
#     "description": "time of first damage in the game",
#     },

#     "tower_kill_by_dire": {
#     "dfname": "",
#     "cumulative": False,
#    "type": "outcome",
#     "description": "a tower was killed by team dire at this timestamp",
#     },
#     "tower_kill_by_radiant": {
#     "dfname": "",
#     "cumulative": False,
#    "type": "outcome",
#     "description": "a tower was killed by team radian at this timestamp",
#     },

#     "barraks_kill_by_dire": {
#     "dfname": "",
#     "cumulative": False,
#    "type": "game",
#     "description": "a barracks was killed by team dire at this timestamp",
#     },
#     "barraks_kill_by_radiant": {
#     "dfname": "",
#     "cumulative": False,
#    "type": "game",
#     "description": "a roshan was killed by team radian at this timestamp. This provides the team with special items that may be useful in fight and for killing towers.",
#     },
#     "roshan_kill_by_dire": {
#     "dfname": "",
#     "cumulative": False,
#    "type": "game",
#     "description": "a barracks was killed by team dire at this timestamp",
#     },
#     "roshan_kill_by_radiant": {
#     "dfname": "",
#     "cumulative": False,
#    "type": "game",
#     "description": "a roshan was killed by team radian at this timestamp. This provides the team with special items that may be useful in fight and for killing towers.",
#     },

#     "event_counts": {
#     "dfnames": ["count_tower_kill_by_dire_1_lag",
#     "count_tower_kill_by_radiant_1_lag",
#     "count_barraks_kill_by_dire_1_lag",
#     "count_barraks_kill_by_radiant_1_lag",
#     "count_roshan_kill_by_dire_1_lag",
#     "count_roshan_kill_by_radiant_1_lag"],
#     "cumulative": True,
#     "type": "game",
#     "description": "these variables show what events have already occured in the game",
#     },
#     "": {
#     "dfname": "",
#     "cumulative": False,
#    "type": "player",
#     "description": "",
#     },
}

def get_all_columns(oneHot=False, raw=True, embedding=True, embeddingConfig={"abilityEmbeddings": True, "heroEmbeddings": True}):
    allCols = [
    'time',
    'PLAYER_0_gold',
    'PLAYER_0_lh',
    'PLAYER_0_xp',
    'PLAYER_0_upgrade_ability_1',
    'PLAYER_0_upgrade_ability_2',
    'PLAYER_0_upgrade_ability_3',
    'PLAYER_0_upgrade_ability_4',
    'PLAYER_0_upgrade_ability_5',
    'PLAYER_0_upgrade_ability_level_1',
    'PLAYER_0_upgrade_ability_level_2',
    'PLAYER_0_upgrade_ability_level_3',
    'PLAYER_0_upgrade_ability_level_4',
    'PLAYER_0_upgrade_ability_level_5',
    'PLAYER_0_hero_id',
    'PLAYER_1_gold',
    'PLAYER_1_lh',
    'PLAYER_1_xp',
    'PLAYER_1_upgrade_ability_1',
    'PLAYER_1_upgrade_ability_2',
    'PLAYER_1_upgrade_ability_3',
    'PLAYER_1_upgrade_ability_4',
    'PLAYER_1_upgrade_ability_5',
    'PLAYER_1_upgrade_ability_level_1',
    'PLAYER_1_upgrade_ability_level_2',
    'PLAYER_1_upgrade_ability_level_3',
    'PLAYER_1_upgrade_ability_level_4',
    'PLAYER_1_upgrade_ability_level_5',
    'PLAYER_1_hero_id',
    'PLAYER_2_gold',
    'PLAYER_2_lh',
    'PLAYER_2_xp',
    'PLAYER_2_upgrade_ability_1',
    'PLAYER_2_upgrade_ability_2',
    'PLAYER_2_upgrade_ability_3',
    'PLAYER_2_upgrade_ability_4',
    'PLAYER_2_upgrade_ability_5',
    'PLAYER_2_upgrade_ability_level_1',
    'PLAYER_2_upgrade_ability_level_2',
    'PLAYER_2_upgrade_ability_level_3',
    'PLAYER_2_upgrade_ability_level_4',
    'PLAYER_2_upgrade_ability_level_5',
    'PLAYER_2_hero_id',
    'PLAYER_3_gold',
    'PLAYER_3_lh',
    'PLAYER_3_xp',
    'PLAYER_3_upgrade_ability_1',
    'PLAYER_3_upgrade_ability_2',
    'PLAYER_3_upgrade_ability_3',
    'PLAYER_3_upgrade_ability_4',
    'PLAYER_3_upgrade_ability_5',
    'PLAYER_3_upgrade_ability_level_1',
    'PLAYER_3_upgrade_ability_level_2',
    'PLAYER_3_upgrade_ability_level_3',
    'PLAYER_3_upgrade_ability_level_4',
    'PLAYER_3_upgrade_ability_level_5',
    'PLAYER_3_hero_id',
    'PLAYER_4_gold',
    'PLAYER_4_lh',
    'PLAYER_4_xp',
    'PLAYER_4_upgrade_ability_1',
    'PLAYER_4_upgrade_ability_2',
    'PLAYER_4_upgrade_ability_3',
    'PLAYER_4_upgrade_ability_4',
    'PLAYER_4_upgrade_ability_5',
    'PLAYER_4_upgrade_ability_level_1',
    'PLAYER_4_upgrade_ability_level_2',
    'PLAYER_4_upgrade_ability_level_3',
    'PLAYER_4_upgrade_ability_level_4',
    'PLAYER_4_upgrade_ability_level_5',
    'PLAYER_4_hero_id',
    'PLAYER_128_gold',
    'PLAYER_128_lh',
    'PLAYER_128_xp',
    'PLAYER_128_upgrade_ability_1',
    'PLAYER_128_upgrade_ability_2',
    'PLAYER_128_upgrade_ability_3',
    'PLAYER_128_upgrade_ability_4',
    'PLAYER_128_upgrade_ability_5',
    'PLAYER_128_upgrade_ability_level_1',
    'PLAYER_128_upgrade_ability_level_2',
    'PLAYER_128_upgrade_ability_level_3',
    'PLAYER_128_upgrade_ability_level_4',
    'PLAYER_128_upgrade_ability_level_5',
    'PLAYER_128_hero_id',
    'PLAYER_129_gold',
    'PLAYER_129_lh',
    'PLAYER_129_xp',
    'PLAYER_129_upgrade_ability_1',
    'PLAYER_129_upgrade_ability_2',
    'PLAYER_129_upgrade_ability_3',
    'PLAYER_129_upgrade_ability_4',
    'PLAYER_129_upgrade_ability_5',
    'PLAYER_129_upgrade_ability_level_1',
    'PLAYER_129_upgrade_ability_level_2',
    'PLAYER_129_upgrade_ability_level_3',
    'PLAYER_129_upgrade_ability_level_4',
    'PLAYER_129_upgrade_ability_level_5',
    'PLAYER_129_hero_id',
    'PLAYER_130_gold',
    'PLAYER_130_lh',
    'PLAYER_130_xp',
    'PLAYER_130_upgrade_ability_1',
    'PLAYER_130_upgrade_ability_2',
    'PLAYER_130_upgrade_ability_3',
    'PLAYER_130_upgrade_ability_4',
    'PLAYER_130_upgrade_ability_5',
    'PLAYER_130_upgrade_ability_level_1',
    'PLAYER_130_upgrade_ability_level_2',
    'PLAYER_130_upgrade_ability_level_3',
    'PLAYER_130_upgrade_ability_level_4',
    'PLAYER_130_upgrade_ability_level_5',
    'PLAYER_130_hero_id',
    'PLAYER_131_gold',
    'PLAYER_131_lh',
    'PLAYER_131_xp',
    'PLAYER_131_upgrade_ability_1',
    'PLAYER_131_upgrade_ability_2',
    'PLAYER_131_upgrade_ability_3',
    'PLAYER_131_upgrade_ability_4',
    'PLAYER_131_upgrade_ability_5',
    'PLAYER_131_upgrade_ability_level_1',
    'PLAYER_131_upgrade_ability_level_2',
    'PLAYER_131_upgrade_ability_level_3',
    'PLAYER_131_upgrade_ability_level_4',
    'PLAYER_131_upgrade_ability_level_5',
    'PLAYER_131_hero_id',
    'PLAYER_132_gold',
    'PLAYER_132_lh',
    'PLAYER_132_xp',
    'PLAYER_132_upgrade_ability_1',
    'PLAYER_132_upgrade_ability_2',
    'PLAYER_132_upgrade_ability_3',
    'PLAYER_132_upgrade_ability_4',
    'PLAYER_132_upgrade_ability_5',
    'PLAYER_132_upgrade_ability_level_1',
    'PLAYER_132_upgrade_ability_level_2',
    'PLAYER_132_upgrade_ability_level_3',
    'PLAYER_132_upgrade_ability_level_4',
    'PLAYER_132_upgrade_ability_level_5',
    'PLAYER_132_hero_id',
    'active_team_fight',
    'team_fight_deaths_per_s',
    'team_fight_deaths_smoothed',
    'team_fight_damage_smoothed',
    'team_fight_damage_per_s',
    'radiant_t1_top_in_range_team_fight',
    'radiant_t1_mid_in_range_team_fight',
    'radiant_t1_bottom_in_range_team_fight',
    'radiant_t2_top_in_range_team_fight',
    'radiant_t2_mid_in_range_team_fight',
    'radiant_t2_bottom_in_range_team_fight',
    'radiant_t3_top_in_range_team_fight',
    'radiant_t3_mid_in_range_team_fight',
    'radiant_t3_bottom_in_range_team_fight',
    'radiant_t4_base_in_range_team_fight',
    'dire_t1_top_in_range_team_fight',
    'dire_t1_mid_in_range_team_fight',
    'dire_t1_bottom_in_range_team_fight',
    'dire_t2_top_in_range_team_fight',
    'dire_t2_mid_in_range_team_fight',
    'dire_t2_bottom_in_range_team_fight',
    'dire_t3_top_in_range_team_fight',
    'dire_t3_mid_in_range_team_fight',
    'dire_t3_bottom_in_range_team_fight',
    'dire_t4_base_in_range_team_fight',
    # 'league_id',
    # 'lobby_type',
    'human_players',
    # 'game_mode',
    'first_blood_time',
    # 'engine',
    'tower_kill_by_dire',
    'tower_kill_by_radiant',
    'barraks_kill_by_dire',
    'barraks_kill_by_radiant',
    'roshan_kill_by_dire',
    'roshan_kill_by_radiant',
    'count_tower_kill_by_dire_1_lag',
    'count_tower_kill_by_radiant_1_lag',
    'count_barraks_kill_by_dire_1_lag',
    'count_barraks_kill_by_radiant_1_lag',
    'count_roshan_kill_by_dire_1_lag',
    'count_roshan_kill_by_radiant_1_lag'
    ]
    if oneHot:
        raise Exception('not implemented')
    elif raw:
        allCols.extend(
            get_category_columns()
        )
    if not embedding or not embeddingConfig.get("abilityEmbeddings", False):
        allCols = [c for c in allCols if not ('upgrade_ability' in c and 'level' not in c)]
    if not embedding or not embeddingConfig.get("heroEmbeddings", False):   
        allCols = [c for c in allCols if not ('hero_id' in c)]

    
    return allCols

def remove_embedding_columns(cols):
    return [c for c in cols if str(c) not in get_embedding_columns()]

def make_one_hot_columns(df, oneHot=False, embedding=True, embeddingConfig = {}):
    if not embedding:
        df = df.drop(columns=get_embedding_columns())
    else:
        abilities, heros = _get_embedding_columns()
        if not embeddingConfig.get("abilityEmbeddings", False):
            df = df.drop(columns=abilities)
        if not embeddingConfig.get("heroEmbeddings", False):
            df = df.drop(columns=heros)
    

    if oneHot:
        # TODO: replace category cols with one hot encoding
        raise Exception('not implemented')
    else:
        return df.drop(columns=get_category_columns())

def get_category_columns():
    return ['league_id','lobby_type', 'game_mode', 'engine']

def _get_embedding_columns():
    return (
        [col for col in get_all_columns() if 'upgrade_ability' in str(col) and 'level' not in str(col)], 
        [c for c in get_all_columns() if 'hero_id' in str(c)]
    )
def get_embedding_columns():
    abilities, heros = _get_embedding_columns()
    abilities.extend(heros)
    return abilities

def get_embedding_columns_in_df(df):
    return [c for c in df.columns if str(c) in get_embedding_columns()]

def get_embedding_dimensions(embeddingConfig = {}):
    abilities, heros = _get_embedding_columns()
    dims = []
    if embeddingConfig.get("heroEmbeddings", False):
        dims.append(
            {
                "feature_names": heros,
                "num_categories": 124,
                # "embedding_dim": 20
            }
        )
    if embeddingConfig.get("abilityEmbeddings", False):
        dims.append(
             {
            "feature_names": abilities,
            "num_categories": 2095,
            # "embedding_dim": 20
        }
        )
    return dims

def get_non_numeric_columns():
    bool_col = [col for col in get_all_columns() if 'kill_by' in str(col) or 'in_range' in str(col)]
    bool_col.extend(['active_team_fight'])
    count_col = [c for c in get_all_columns() if 'count' in str(c)]
    count_col.extend(["human_players"])
    categories = get_embedding_columns()
    categories.extend(get_category_columns())

    result = []
    result.extend(bool_col)
    result.extend(count_col)
    result.extend(categories)

    return result

class DataDictionary:
    def __init__(self, fields):
        self.fields = {}
        for k, v in fields.items():
            if v["dfname"] is None:
                v["dfname"] = k
            self.fields[k] = DataDictionaryField(v)
        
        self.time = self.fields['time'].dfname
    
    def col(self, k):
        return self.fields[k].dfname

class DataDictionaryField:
    def __init__(self, field):
        self.dfname = field["dfname"]
        self.cumulative = bool(field["cumulative"])
        self.type = field['type']
        self.description = field['description']


f = DataDictionary(data_dictionary_fields)


TS_INTERVAL = 60 # default interval for binning observations

def get_row_t(time):
    return int(math.ceil(time / TS_INTERVAL))

def get_tower_locations():
    """
    https://dota2.fandom.com/wiki/Map
    https://developer.valvesoftware.com/wiki/Dota_2_Workshop_Tools/Level_Design/Dota_Map
    """
    return [
        # Radiant Towers
        # Tier 1 Towers:
        # Top: 
        (4860, 10860),
        # Mid:
        (6280, 6150),
        # Bottom:
        (10860, 4860),
        # Tier 2 Towers:
        # Top:
        (5760, 8340),
        # Mid:
        (5510, 4960),
        # Bottom:
        (8340, 5760),
        # Tier 3 Towers:
        # Top: 
        (6180, 6620),
        # Mid: 
        (4900, 3900),
        # Bottom:
        (6620, 6180),
        # Tier 4 Towers (Base):
        # First: 
        (4475, 3980),
        # Second:
        (3980, 4475),

        #Dire Towers
        # Tier 1 Towers:
        # Top: 
        (4860, 10860),
        # Mid: 
        (10480, 10480),
        # Bottom: 
        (10860, 4860),
        # Tier 2 Towers:
        # Top: 
        (8200, 8460),
        # Mid: 
        (10480, 7940),
        # Bottom: 
        (8460, 8200),
        # Tier 3 Towers:
        # Top: 
        (6300, 8450),
        # Mid:
        (9000, 9000),
        # Bottom: 
        (8450, 6300),
        # Tier 4 Towers (Base):
        # First: 
        (10880, 10540),
        # Second: 
        (10540, 10880),
    ]

# will map item ids to integers 1...n, can be used for embedding input or one hot encoding. There are 1000+ items
def get_item_embedding_map():
    with open('./opendota_constants/items.json', 'r') as f:
        data = json.load(f)
        return {item_data['id']: idx for idx, item_data in enumerate(data['itemdata'].values())}
    
def get_item_embedding_input(map, item_id):
    if item_id is not None and item_id in map:
        return map[item_id] + 1
    else:
        return 0 # unknown or empty is 
    
# will map abilities to integers 1...n, can be used for embedding input or one hot encoding. There are 1000+ items
def get_abilities_embedding_map():
    with open('./opendota_constants/abilities.json', 'r') as f:
        data = json.load(f)
        return {ability: idx for idx, ability in enumerate(data.keys())}
    
# abilities are stored as numbers, and the map does not include the numbers
def get_abilities_embedding_input(map, ability):
    return ability

    # if ability in map:
    #     return map[ability]
    # else:
    #     return len(map) #create a new i for unknown

def get_tower_names():
    return [
        # Radiant Towers
        "radiant_t1_top", 
        "radiant_t1_mid", 
        "radiant_t1_bottom",
        "radiant_t2_top", 
        "radiant_t2_mid", 
        "radiant_t2_bottom",
        "radiant_t3_top", 
        "radiant_t3_mid", 
        "radiant_t3_bottom",
        "radiant_t4_base", 
        "radiant_t4_base",

        # Dire Towers
        "dire_t1_top", 
        "dire_t1_mid", 
        "dire_t1_bottom",
        "dire_t2_top", 
        "dire_t2_mid", 
        "dire_t2_bottom",
        "dire_t3_top", 
        "dire_t3_mid", 
        "dire_t3_bottom",
        "dire_t4_base", 
        "dire_t4_base"
    ]

def get_allowed_game_modes():
    """
    https://github.com/odota/core/blob/7583e3b60c0d37e40aece80cb0ca891ec96da2db/json/game_mode.json
    """
    pass


def get_last_event_timestamp(match_data):
    """
    Get the timestamp of the last recorded event in a match. This is necessary because the duration in the data is sometimes shorter than the recorded data
    
    Args:
        match_data: OpenDota match data
        
    Returns:
        The timestamp (in seconds) of the last recorded event
    """
    timestamps = []
    
    try:
        # Check objectives (tower kills, barracks, etc.)
        if 'objectives' in match_data and match_data['objectives']:
            for objective in match_data.get('objectives', []) or []:
                if 'time' in objective:
                    timestamps.append(objective['time'])
    except:
        pass
    
    try:
        # Check ability upgrades for all players
        if 'players' in match_data and match_data['players']:
            for player in match_data.get('players', []) or []:
                for upgrade in player.get('ability_upgrades', []) or []:
                    if 'time' in upgrade:
                        timestamps.append(upgrade['time'])
        
            # Check player.time if it exists (varies by API version)
            for player in match_data.get('players', []) or []:
                if hasattr(player, 'time') and player.time:
                    timestamps.append(player.time)
    except:
        pass
    
    try:
        # Check gold advantage arrays
        # These are usually recorded at regular intervals
        if 'radiant_gold_adv' in match_data and match_data['radiant_gold_adv']:
            # Each entry represents a minute, so multiply by 60 for seconds
            last_index = len(match_data['radiant_gold_adv']) - 1
            timestamps.append(last_index * 60)
    except:
        pass
    
    try:
        # Check end time of the last teamfight
        if 'teamfights' in match_data and match_data['teamfights']:
            last_teamfight = match_data['teamfights'][-1]
            if 'end' in last_teamfight:
                timestamps.append(last_teamfight['end'])
    except:
        pass
    
    # Return the maximum timestamp if we found any events
    if timestamps:
        return max(timestamps)
    else:
        # Fallback to the reported duration if no events were found
        if 'duration' in match_data:
            return match_data.get('duration', 0)
        return 0

# player fields include the player slot
def p_col(player_slot, colname):
     return f"PLAYER_{player_slot}_{colname}"

def p_cols(player_slot, colnames):
     return [p_col(player_slot, cn) for cn in colnames]

def df_cols(colnames, cols):
     return { colnames[i]: cols[i] for i in range(len(colnames)) }

def centroid(coordinates):
    return (
        sum(int(x) for x, _ in coordinates) / len(coordinates),
        sum(int(y) for _, y in coordinates) / len(coordinates)
    )

def find_closest_coordinate(c, coordinates):  
    distances = [math.dist(c, coord) for coord in coordinates]
    closest_index = distances.index(min(distances))
    return closest_index, distances[closest_index]

def unix_intervals(start, end, freq):
    intervals = pd.date_range(
        start=pd.to_datetime(start, unit='s'), 
        end=pd.to_datetime(end, unit='s'), 
        freq=f'{freq}s'
    )
    return intervals.astype(np.int64) // 10**9

def reindex(df, times, method='ffill', value=None, limit=None):
    if df[f.time].duplicated().any():
        df = df.drop_duplicates(subset=[f.time], keep='last')

    result = df.set_index(f.time)
    
    result = result.reindex(pd.Index(times, name=f.time), columns=result.columns)
    if value:
        result = result.fillna(value)
    elif method == 'ffill':
        result = result.ffill(limit=limit) #note ffill will leave NaNs before the first value from input
    elif method == 'ffill_custom':
        result = result.reset_index()
        ffill_custom(df, result)
        return result
    elif method == "bfill":
        result = result.bfill(limit=limit)
    result = result.reset_index()
    return result

def ffill_custom(dffrom, dfto):
    ti = 0
    for ai in range(len(dffrom[f.time])):
        t = dffrom.loc[ai, f.time]
        while ti < dfto.shape[0] and dfto.loc[ti, f.time] < t:
            ti+=1
        t2 = dfto.loc[ti - 1, f.time]
        dfto.iloc[ti - 1] = dffrom.iloc[ai]
        dfto.loc[ti - 1, f.time] = t2

def transform_json_to_ts(data):
        missing_data = {}

        # first get all time series data
        # mStart = data["start_time"]
        mEnd = get_last_event_timestamp(data)

        if mEnd == 0:
            missing_data['duration'] = True
            missing_data['any'] = True
            missing_data['error'] = "cannot find duration"
            return pd.DataFrame({ f.time: []}), missing_data

        times = unix_intervals(start=0, end=mEnd, freq=TS_INTERVAL)

        df = pd.DataFrame({ f.time: times, })

        if not "players" in data:
            missing_data["players"] = True
        else:
            try:
                for p in data["players"]:
                    slot = str(p.get("player_slot", "unknown")) or "unknown"
                    if slot == "unknown":
                        raise Exception("player slot is unknown")
                    # add time columns

                    if not (p["times"] and p["gold_t"] and p["lh_t"] and p["xp_t"]):
                        missing_data["players_times"] = True
                    else:

                        pdf = pd.DataFrame(df_cols(
                            [f.time] + p_cols(slot, [f.col('gold'), f.col('lh'), f.col('xp')]),
                            [p["times"], p["gold_t"], p["lh_t"], p["xp_t"]],
                        ))

                        # next, add time samples for full lenght of game if not already
                        # reindex with ffill
                        pdf = reindex(pdf, times, 'ffill')
                        
                        
                        df = df.merge(pdf, on=f.time, how='inner')

                    # then, add current ability_levels into the dataset (how many abilities total?)
                    # turn into ts, reindex with ffill
                    # p["ability_upgrades"]

                    # first order abilities by code
                    # TODO: create consistent order for each hero by reading in all available options
                    if not ("ability_upgrades" in p and p["ability_upgrades"]):
                        missing_data["ability_upgrades"] = True        
                    else:
                        try:
                            upgrades = list(p["ability_upgrades"])
                            upgrades.sort(key=lambda u: u["time"])

                            recent_abilities = []

                            dfrows = []
                            rowtimes = {}
                            for u in upgrades:
                                if not ("ability" in u and "time" in u):
                                    missing_data["ability_upgrades.properties"] = True 
                                else:
                                    uTime = u["time"]
                                    rowI = len(dfrows) - 1
                                    if uTime in rowtimes:
                                        rowI = rowtimes[uTime]
                                    else:
                                        row = [math.nan] * 11
                                        row[0] = u["time"]
                                        dfrows.append(row)
                                        rowtimes[uTime] = rowI

                                    # Add 1 because 0 will indicate no ability upgrade at this timestamp.
                                    if u["ability"]:
                                        if u["ability"] in recent_abilities:
                                            # move to front of list
                                            recent_abilities.remove(u["ability"])
                                        else:
                                            recent_abilities = recent_abilities[0:4]
                                        
                                        recent_abilities.append(u["ability"])

                                    if u["ability"] and u["ability"] in recent_abilities:
                                        cndx = 1 + recent_abilities.index(u["ability"])
                                        # print(cndx)
                                        if cndx > 5 or cndx < 1:
                                            raise Exception("cndx is >5 or <1")
                                        dfrows[rowI][cndx] = int(u["ability"]) + 1

                                        if not ("level" in u):
                                            missing_data["ability_upgrades.level"] = True
                                        else:
                                            cndx = 6 + recent_abilities.index(u["ability"])
                                            # print(cndx)
                                            if cndx > 10 or cndx < 6:
                                                raise Exception("cndx2 is >10 or <6")
                                            dfrows[rowI][cndx] = u["level"]

                        

                            padf = pd.DataFrame(
                                dfrows,
                                columns=[f.time] + p_cols(slot, [f.col('upgrade_ability_1'), f.col('upgrade_ability_2'), f.col('upgrade_ability_3'), f.col('upgrade_ability_4'), f.col('upgrade_ability_5'), f.col('upgrade_ability_level_1'), f.col('upgrade_ability_level_2'), f.col('upgrade_ability_level_3'), f.col('upgrade_ability_level_4'), f.col('upgrade_ability_level_5')]),
                            )

                            # print(padf[['time','PLAYER_0_upgrade_ability_1']])

                            padf = reindex(padf, times, method='ffill_custom')

                            padf = padf.fillna(0.0)
                            # join to df
                            df = df.merge(padf, on=f.time, how='inner')
                        except Exception as e:
                            raise Exception(f"error processing ability upgrades: {str(e)}")

                    # print(patdf[['time','PLAYER_0_upgrade_ability_1']])

                    
                    # TODO: then add purchases () - difficult because thousands? of items available for purchase. Player can only carry a few in 5 hero slots and a few backpack slots
                    # data includes items 0-5, but no idea when in game these are recorded and can be changed often.
                    # TODO: then add (npc) kills from kills_log - kills listed here have the name of hero - must be another hero in the game. Usually no duplicate heros in a match except a few game modes like turbo (20), custom (16), event (18)
                    # turn into ts, reindex with ffill

                    # TODO: add hero_id, leaver_status
                    df[p_col(slot, "hero_id")] = p["hero_id"]
                    # df[p_col(slot, "hero_id")] = p["hero_id"]

                    # TODO estimated position for later
                    # TODO: Would be nice to calculate experience score or cumulative stats per account
            except Exception as e:
                missing_data["players"] = True
                missing_data["players_error"] = str(e)
                missing_data["stack_trace"] = get_stacktrace()
                missing_data["slot"] = slot # type: ignore


        
        
        # now add data from teamfights

        # default values
        df['active_team_fight'] = False
        df['team_fight_deaths_per_s'] = 0.0
        df['team_fight_deaths_smoothed'] = 0.0
        df['team_fight_damage_smoothed'] = 0.0
        df['team_fight_damage_per_s'] = 0.0

        # add rows for towers in range
        tower_dist_cols = [f"{tname}_in_range_team_fight" for tname in get_tower_names()]
        df[tower_dist_cols] = False

        if not ("teamfights" in data and data["teamfights"]):
            missing_data["teamfights"] = True
        else:
            try:
                for tf in data["teamfights"]:
                    positions = []
                    # deathsDire = 0
                    # deaths_
                    total_damage = 0.0

                    if "players" in tf:
                        for p in tf["players"]:
                            # if p["deaths"]:
                            #     deaths.append(p["deaths"])
                            if p["deaths_pos"]:
                                for x, yobj in p["deaths_pos"].items():
                                    for y in yobj.keys():
                                        positions.append((int(x), int(y)))
                            if p["damage"]:
                                total_damage += p["damage"]
                            # if p["killed"]:
                            #     for killed_hero in p["killed"].keys():
                            #         heroName = killed_hero.removeprefix("npc_dota_hero_")
                            #         if isDireHero(heroName):
                            #             deathsDire += 1
                            #         else:
                            #             pass # TODO

                            # TODO: more advanced individual tracking:
                            # identify each player in the teamfight
                                # look at xp delta, xp start, xp end around start end times
                                #   if multiple matches look at gold delta
                                #       if multiple matches, look at ability_uses
                                # add deaths, hero kills at time of last death

                            # Would be nice to include abilities used, but too many abilities

                
                    if "start" in tf and "end" in tf and "deaths" in tf and "last_death" in tf:
                        tfStart = int(tf.get("start", 0)) or 0
                        tfEnd = int(tf.get("end", 0)) or 0
                        tfDeaths = int(tf.get("deaths", 0)) or 0
                        tfLastDeath = int(tf.get("last_death", 0)) or 0
                        tfDuration = tfEnd - tfStart

                        row_start = get_row_t(tfStart)
                        row_end = get_row_t(tfEnd)
                        row_last_death = get_row_t(tfLastDeath)
                        
                        if row_end - row_start <= 1:
                            df.loc[row_end, 'team_fight_damage_smoothed'] = total_damage
                            df.loc[row_end, 'active_team_fight'] = True
                            df.loc[row_end, 'team_fight_deaths_per_s'] = tfDeaths / tfDuration # alternate
                            df.loc[row_end, 'team_fight_damage_per_s'] = total_damage / tfDuration # alternate
                        else:
                            df.loc[row_start:row_end, 'team_fight_damage_smoothed'] = np.linspace(0.0, total_damage, row_end - row_start + 1) # assume damage is accrued constantly
                            df.loc[row_start:row_end, 'active_team_fight'] = True
                            df.loc[row_start:row_end, 'team_fight_deaths_per_s'] = tfDeaths / tfDuration # alternate
                            df.loc[row_start:row_end, 'team_fight_damage_per_s'] = total_damage / tfDuration # alternate

                        if row_last_death - row_start <=1:
                            df.loc[row_last_death, 'team_fight_deaths_smoothed'] = tfDeaths
                        else:
                            df.loc[row_start:row_last_death, 'team_fight_deaths_smoothed'] = np.linspace(0.0, tfDeaths, row_end - row_start + 1) # assume deaths are accrued constantly


                    # if location is known, find nearest tower and record proximity
                    if len(positions):
                        estimatedPosition = centroid(positions)
                        closest_tower_i, closest_tower_dist = find_closest_coordinate(estimatedPosition, get_tower_locations()) #TODO: get tower positions
                        # distance cutoff of 844 + 350
                        # Tower Attack Range: 700 units
                        # Tower Acquisition Range: 850 units
                        # Tower Collision Size: 144 units (radius)
                        # Melee Hero Range: 150
                        # Ranged Heros: up to 700
                        # centroid of teamfight between two ranged heros can be up to 350 units beyond one hero's range of attack on the tower
                        # Sources: https://dota2.fandom.com/wiki/Buildings#Towers, https://liquipedia.net/dota2/Buildings, https://dota2.gamepedia.com/Buildings#Towers
                        if closest_tower_dist <= 1194:
                            tower_col = tower_dist_cols[closest_tower_i]
                            row_start = get_row_t(tfStart) # type: ignore
                            row_end = get_row_t(tfEnd) # type: ignore
                            df.loc[row_start:row_end, tower_col] = True
            except Exception as e:
                missing_data["teamfights"] = True
                missing_data["teamfights_error"] = str(e)
                missing_data["stack_trace"] = get_stacktrace()
            

        # Add team columns for xp
        # These are often missing so we will ignore them
        # if not ("radiant_gold_adv" in data and data["radiant_gold_adv"]):
        #     missing_data["radiant_gold_adv"] = True
        # else:
        #     rad_gold_adv = data["radiant_gold_adv"]
        #     rad_gold_adv_times = unix_intervals(start=0, end=TS_INTERVAL * (len(rad_gold_adv) - 1), freq=TS_INTERVAL)
        #     tgdf = pd.DataFrame({ f.time: rad_gold_adv_times, "radiant_gold_adv": rad_gold_adv })
        #     tgdf = reindex(tgdf, times, 'ffill')

        #     df = df.merge(tgdf, on=f.time, how='inner')

        # if not ("radiant_xp_adv" in data and data["radiant_xp_adv"]):
        #     missing_data["radiant_xp_adv"] = True
        # else:
        #     rad_xp_adv = data["radiant_xp_adv"]
        #     rad_xp_adv_times = unix_intervals(start=0, end=TS_INTERVAL * (len(rad_xp_adv) - 1), freq=TS_INTERVAL)
        #     txdf = pd.DataFrame({ f.time: rad_xp_adv_times, "radiant_xp_adv": rad_xp_adv })
        #     txdf = reindex(txdf, times, 'ffill')

        #     df = df.merge(txdf, on=f.time, how='inner')


        # Add game columns: leagueid, lobby_type?, human_players, game_mode

        try:
            df['league_id'] = data["leagueid"]
            df['lobby_type'] = data["lobby_type"]
            df['human_players'] = data["human_players"]
            df['game_mode'] = data["game_mode"]
            df['first_blood_time'] = data["first_blood_time"]
            df['engine'] = data["engine"]
        except Exception as e:
            missing_data["match_fields"] = True
        
        # Add columns for objectives
        # Add columns for team objectives
        df['tower_kill_by_dire'] = False
        df['tower_kill_by_radiant'] = False
        df['barraks_kill_by_dire'] = False
        df['barraks_kill_by_radiant'] = False
        df['roshan_kill_by_dire'] = False
        df['roshan_kill_by_radiant'] = False

        if not ("objectives" in data and data["objectives"]):
            missing_data["objectives"] = True
        else:
            try:
                for o in data["objectives"]:
                    row_o = get_row_t(o["time"])
                    event_col = None
                    if o["subtype"] == 'CHAT_MESSAGE_TOWER_KILL':
                        if o["team"] == 2: # Radiant
                            event_col = 'tower_kill_by_radiant'
                        else:
                            event_col = 'tower_kill_by_dire'
                    elif o["subtype"] == 'CHAT_MESSAGE_ROSHAN_KILL':
                        if o["team"] == 2: # Radiant
                            event_col = 'roshan_kill_by_radiant'
                        else:
                            event_col = 'roshan_kill_by_dire'
                    elif o["subtype"] == 'CHAT_MESSAGE_BARRACKS_KILL':
                        if o["key"] == 2: # Radiant barracks killed
                            event_col = 'barraks_kill_by_dire'
                        else:
                            event_col = 'barraks_kill_by_radiant'
                    
                    if event_col is not None:
                        df.loc[row_o, event_col] = True
            except Exception as e:
                missing_data["objectives"] = True
                missing_data["objectives_error"] = str(e)
                missing_data["stack_trace"] = get_stacktrace()

        # Make time series columns for cumulative objectives achieved before now
        df['count_tower_kill_by_dire_1_lag'] = df['tower_kill_by_dire'].shift(1).cumsum().fillna(0).astype('float32')
        df['count_tower_kill_by_radiant_1_lag'] = df['tower_kill_by_radiant'].shift(1).cumsum().fillna(0).astype('float32')
        df['count_barraks_kill_by_dire_1_lag'] =  df['barraks_kill_by_dire'].shift(1).cumsum().fillna(0).astype('float32')
        df['count_barraks_kill_by_radiant_1_lag'] = df['barraks_kill_by_radiant'].shift(1).cumsum().fillna(0).astype('float32')
        df['count_roshan_kill_by_dire_1_lag'] = df['roshan_kill_by_dire'].shift(1).cumsum().fillna(0).astype('float32')
        df['count_roshan_kill_by_radiant_1_lag'] = df['roshan_kill_by_radiant'].shift(1).cumsum().fillna(0).astype('float32')

        # summarize missing_data
        hasError = False
        hasMissingData = False
        for k, v in missing_data.items():
            if v:
                hasMissingData = True
            if k.endswith("error"):
               hasError = True
        
        if hasError:
            missing_data["error"] = True
        if hasMissingData:
            missing_data["any"] = True


        cols_to_convert = df.select_dtypes(exclude=['bool']).columns
        df[cols_to_convert] = df[cols_to_convert].astype('float32').fillna(0.0)

        bool_cols = df.select_dtypes(include=['bool']).columns
        df[bool_cols] = df[bool_cols].fillna(False)

        # # Check for None values
        # print(df.isnull().sum())

        # # Check for infinite values
        # print(np.isinf(df.select_dtypes(include=[np.number])).sum())


        # for c in df.columns:
        #     try:
        #         print(c)
        #         torch.tensor(df[c].values, dtype=torch.float32)
        #     except Exception as e:
        #         print("Conversion error:", e)
        #         raise e
                        

        return df, missing_data




# todo: save the data in this format so we don't have to calculate again later

# save a version with lags so we don't have to calculate again later - use similar lags as death pred paper, also include leads outcomes

# todo: load and transform, save data, or just load it





def addLags(df,lag_list,col_list,direction = 'lag'):

    if direction == 'lead':
        lag_list = map(lambda x: x*(-1),lag_list)

    arr_lags = list(map(partial(_buildLags,df=df,
                        col_list=col_list,
                        direction = direction),
                lag_list))

    df = pd.concat([df]+arr_lags,axis = 1)

    return df

def _buildLags(lag,df,col_list,direction):

    return df[col_list].shift(lag).add_suffix('_{}_{}'.format(np.abs(lag),direction))



class MatchesDataset(Dataset):
    def __init__(self, matches_list, config):
        self.matches = matches_list
        self.rangeTracker = load_range_tracker('./statistics/normalize_ranges.json')
        # use the same range for all upgrade ability levels
        all_update_ability_level_columns = [col for col in get_all_columns() if 'upgrade_ability_level' in str(col)]
        self.rangeTracker.merge_columns(all_update_ability_level_columns, 'upgrade_ability_level')
        # store columns to normalize
        self.numeric_columns = [col for col in get_all_columns() if col not in get_non_numeric_columns()]
        self.config = config

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        # Collect data for the match - n balanced rows
        match_info = self.matches[idx]

        col_names = get_all_columns(oneHot=self.config["one_hot"],raw=False,embedding=self.config["embedding"], embeddingConfig=self.config["embeddings"])

        valid = True

        try:

            data = json.loads(load_file_from_s3(match_info["key"]))
            df, missing_data = transform_json_to_ts(data)

            if "any" in missing_data and missing_data["any"] is True:
                print(missing_data)
                if self.config.get("skip_all_data_errors", False) is True:
                    valid = False

            for k in missing_data.keys():
                if "_error" in str(k) and missing_data[k] is not None:
                    print(missing_data.get("stack_trace", "") or "")
                    raise Exception(f"error in data shaping: {missing_data[k]}")
                elif missing_data["error"] is True:
                    raise Exception(f"unknown error in data shaping")

        
            y_cols = get_outcome_tower_columns(df)
            df = sample_balanced_rows(df, y_cols)

            df = self.rangeTracker.z_score_normalize(df, columns=self.numeric_columns)

            df = make_one_hot_columns(df, oneHot=self.config["one_hot"], embedding=self.config["embedding"], embeddingConfig=self.config["embeddings"]) # all cols will no longer be raw

            # print(df.head(3))

            # print_full_df(df)

            # print(df[None])


            

            if len(df.columns) != len(col_names):
                print(len(df.columns))
                print(f"all col{len(col_names)}")
                print([col for col in col_names if "ability" in str(col)])
                valid = False
                raise Exception("missing columns in data")
        
        except Exception as e:
            # create a dummy dataset of all -1.
            df = pd.DataFrame(np.full((1, len(col_names)), -1), columns=col_names)
            y_cols = get_outcome_tower_columns(df)
            valid = False
            print(match_info["key"])
            print(e)

        xdf = df.drop(columns=y_cols)
        ydf = df[y_cols]

        if self.config["embedding"] is True:
            xdf = xdf.drop(columns=get_embedding_columns_in_df(xdf))


        # # Get columns with non-supported types
        # unsupported_cols = df.select_dtypes(exclude=['float64', 'float32', 'float16', 'complex64', 'complex128', 
        #                                             'int64', 'int32', 'int16', 'int8', 
        #                                             'uint64', 'uint32', 'uint16', 'uint8', 
        #                                             'bool']).columns.tolist()

        # # Print columns and their types for more details
        # for col in unsupported_cols:
        #     print(f"{col}: {df[col].dtype}")



        # for c in ydf.columns:
        #     try:
        #         print(c)
        #         print(len(ydf[c]))
        #         torch.tensor(ydf[c].values, dtype=torch.float32)
        #     except Exception as e:
        #         print("Conversion error:", e)
        #         raise e


        # print(list(df.columns))
        if len(xdf.columns[xdf.isnull().any()].tolist()) > 0 or len(ydf.columns[ydf.isnull().any()].tolist()) > 0:
            print("NaN data:")
            print(xdf.columns[xdf.isnull().any()].tolist())
            print(ydf.columns[ydf.isnull().any()].tolist())
        
        x = {
            'tensor': torch.tensor(xdf.values.astype(np.float32), dtype=torch.float32),
            'columns': xdf.columns.tolist(),
        }

        if self.config.get("embedding", False):
            x["categorical_features"] = {}
            for feature in get_embedding_dimensions(self.config["embeddings"]):
                for feature_name in feature["feature_names"]:
                    fdf = df[feature_name]
                    x["categorical_features"][feature_name] = torch.tensor(fdf.values.astype(np.int32), dtype=torch.int32)

        y = {
            'tensor': torch.tensor(ydf.values.astype(np.float32), dtype=torch.float32),
            'columns': ydf.columns.tolist()
        }

        return x, y, valid


def is_invalid_data(t):
    torch.all(t == -1)


def collate_batch(batch_tensors):
    x_tensors = []
    x_cols = None
    x_categorical_tensors = {}

    y_tensors = []
    y_cols = None

    for (x, y, valid) in batch_tensors:
        x_cols = x["columns"]
        y_cols = y["columns"]
        if valid is True:
            x_tensors.append(x["tensor"])
            y_tensors.append(y["tensor"])

            if x.get("categorical_features") is not None:
                for feature in x["categorical_features"]:
                    if feature not in x_categorical_tensors:
                        x_categorical_tensors[feature] = []
                    x_categorical_tensors[feature].append(x["categorical_features"][feature])

    for feature in x_categorical_tensors:
        x_categorical_tensors[feature] = torch.cat(x_categorical_tensors[feature], dim=0)
                        

    return {
        "tensor": torch.cat(x_tensors, dim=0),
        "columns": x_cols,
        "valid": len(x_tensors) > 0,
        "count_matches": len(x_tensors),
        "categorical_features": x_categorical_tensors,
    },{
        "tensor": torch.cat(y_tensors, dim=0),
        "columns": y_cols
    }


def create_batch_dataloader(matches_list, batch_size=32, shuffle=True, config={'one_hot': False, 'embedding': False, 'skip_all_data_errors': False}, training_cpu=True):
    dataset = MatchesDataset(matches_list, config)
    
    num_cpus = os.cpu_count()
    if num_cpus:
        if training_cpu:
            num_workers = max(1, num_cpus // 3) # allow more cpu for training
        else:
            num_workers = max(1, num_cpus - 1) # use cpu for loading, gpu for training
    else:
        num_workers = 1

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=collate_batch,
        num_workers=num_workers,  # This enables parallel data loading on n-1 CPUs
        pin_memory=True,  # This speeds up the transfer from CPU to GPU (if using GPU)
        persistent_workers=True,  # Keep workers alive between epochs (reduces startup overhead)
        prefetch_factor=4  # Each worker will prefetch 3 × batch_size samples (can make larger if memory allows)
    )
    return dataloader

def _test_dataloader():
    matches, _, _ = train_evel_test_split()
    
    dataloader = create_batch_dataloader(matches, 2, False)
    
    for i, (x, y) in enumerate(dataloader):
        if i >= 3:
            break
        print(f"Batch {i}:")
        print("\nis valid:\n", x["valid"])
        print("\nColumns:\n", x["columns"])
        print("\nFeatures:\n", x["tensor"][:2])
        print("\nLabels:\n", y["tensor"][:2])
        print("---")


def load_samples(key):
    data = json.loads(load_file_from_s3(key))
    df = transform_json_to_ts(data)
    y_cols = get_outcome_tower_columns(df)
    df = sample_balanced_rows(df, y_cols)
    return df

def load_data_keys():
    with open('./data/all_match_files.json', 'r') as file:
        return json.load(file)

def load_file_from_s3(key):
   s3_client = boto3.client('s3')
   response = s3_client.get_object(Bucket="cs229-dota", Key=key)
   return response['Body'].read().decode('utf-8')

def train_evel_test_split(eval_size = 0.1, test_size = 0.1, allowNoMatches=False):
    """
    Loads match ids and Splits the dataset into a temporal development / test split, and into a random train / eval splits.
    This is designed to simulate training on incoming real-world data.
    """
    data = load_data_keys()
    if allowNoMatches:
        matches = data.values()
    else:
        matches = list(filter(lambda match: "no_tower_kills" not in str(match["key"]), data.values()))

    matches = sorted(matches, key=lambda match: match["time"])

    # test split - last 10% of data
    t_index = math.floor(len(matches) * (1 - test_size))
    test_split = matches[t_index:]

    adjusted_eval_size = len(matches) / t_index  * eval_size # eval_size is relative to full dataset, so must be adjusted after the test_split is removed

    train_split, eval_split, = train_test_split(matches[0:t_index], test_size=adjusted_eval_size, random_state=42)

    return train_split, eval_split, test_split


def get_outcome_tower_columns(df):
    """
    Gets all columns tracking tower kills that are leading or current
    """
    return _get_outcome_tower_columns(list(df.columns))

def _get_outcome_tower_columns(cols):
    return [col for col in cols if ('tower_kill_by_dire' in str(col) or 'tower_kill_by_radiant' in str(col)) and not 'lag' in str(col)]


def get_features_in_out(oneHot=False, embedding=False, embeddingConfig={}):
    allCols = get_all_columns(oneHot=oneHot, embedding=embedding, raw=False, embeddingConfig=embeddingConfig)
    print(len(allCols))
    outcomeCols = _get_outcome_tower_columns(get_all_columns())
    return [c for c in allCols if not c in outcomeCols], outcomeCols 

def sample_balanced_rows(df, outcome_columns):
    """
    Sample all rows when towers were killed.
    Sample equivalent amount of rows when towers were not killed.
    """
    tower_kill_rows = df[df[outcome_columns].any(axis=1)]

    if len(tower_kill_rows) == 0:
        # in this case take 16 rows
        sample_size = min(16, len(df))
        return df.sample(n=sample_size)

    sample_size = min(len(tower_kill_rows), len(df) - len(tower_kill_rows))
    no_tower_kill_rows = df[ ~df[outcome_columns].any(axis=1)].sample(n=sample_size)
    return pd.concat([tower_kill_rows, no_tower_kill_rows])


def test():
    # file = './dataExamples/teamfights.json'
    file = './dataExamples/1827780053_1620896617.json'

    with open(file, 'r') as file:
        data = json.load(file)
        df, missing_data = transform_json_to_ts(data)
        # print(list(df.columns))
        # print(df.shape)
        # print(df.tail(15))
        print(missing_data)

        # print_full_df(df)

        # Method 1: Columns with NaN
        # columns_with_nan = df.columns[df.isnull().any()].tolist()
        # print(columns_with_nan)

        # outcome_columns = get_outcome_tower_columns(df)
        # print(outcome_columns)
        # print(sample_balanced_rows(df, outcome_columns))

    # train, eval, test_split = train_evel_test_split()
    # print(train[0:5])
    # print(len(train))
    # print(len(eval))
    # print(len(test_split))

    # _test_dataloader()

if __name__ == "__main__":
    test()
