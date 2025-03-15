# cs229-dota

Data Dictionary  
Fields with type “player” have 10 features, one for each player

| Variable Name | Category | Description |
| :---- | :---- | :---- |
| time | game | Time since the start of the game, in seconds |
| gold | player | Gold available for purchases, earned while playing the game |
| lh | player | Continuous measurement of health (last hits) |
| xp | player | Experience points gained by fighting in the game |
| upgrade\_ability\_1 | player | Ability that was upgraded, will be NaN at timestamps with no upgrade. There are 2095 abilities, each represented by a four digit code |
| upgrade\_ability\_2 | player | Ability that was upgraded, will be NaN at timestamps with no upgrade |
| upgrade\_ability\_3 | player | Ability that was upgraded, will be NaN at timestamps with no upgrade |
| upgrade\_ability\_4 | player | Ability that was upgraded, will be NaN at timestamps with no upgrade |
| upgrade\_ability\_5 | player | Ability that was upgraded, will be NaN at timestamps with no upgrade |
| upgrade\_ability\_level\_1 | player | New level of ability that was upgraded |
| upgrade\_ability\_level\_2 | player | New level of ability that was upgraded |
| upgrade\_ability\_level\_3 | player | New level of ability that was upgraded |
| upgrade\_ability\_level\_4 | player | New level of ability that was upgraded |
| upgrade\_ability\_level\_5 | player | New level of ability that was upgraded |
| game\_mode | player | Game mode, see [DOTA Constants](https://github.com/odota/dotaconstants/blob/master/build/game_mode.json) |
| active\_team\_fight | game | Are there currently any active team fights. Team fights are identified by the game when players are close together and dealing damage |
| team\_fight\_deaths\_per\_s | teamfight | A measurement of the intensity of the team fight. This statistic is 0 until the first death in a team fight and returns to zero after the fight ends |
| team\_fight\_deaths\_smoothed | teamfight | The raw data does not include the time of deaths, so this attempts to approximate similar data by assuming deaths occur at a constant rate. This static linearly increases during a team fight from the first death until the last death |
| team\_fight\_damage\_smoothed | teamfight | The raw data does not include damage with timestamps, so this approximates it by linearly smoothing the total damage in the fight over time |
| team\_fight\_damage\_per\_s | teamfight | This is a measure of the intensity of a team fight |
| human\_players | game | Count of human players in the game. Range 1 to 10 |
| first\_blood\_time | game | Time of first damage in the game |
| tower\_kill\_by\_dire | outcome | A tower was killed by team dire at this timestamp |
| tower\_kill\_by\_radiant | outcome | A tower was killed by team radiant at this timestamp |
| barraks\_kill\_by\_dire | game | A barracks was killed by team dire at this timestamp |
| barraks\_kill\_by\_radiant | game | A barracks was killed by team radiant at this timestamp |
| roshan\_kill\_by\_dire | game | A roshan was killed by team dire at this timestamp. This provides the team with special items that may be useful in fight and for killing towers |
| roshan\_kill\_by\_radiant | game | A roshan was killed by team radiant at this timestamp. This provides the team with special items that may be useful in fight and for killing towers |
| radiant\_t1\_top\_in\_range\_team\_fight | teamfight | Team fight is within range (844 units) of Radiant's tier 1 top tower |
| radiant\_t1\_mid\_in\_range\_team\_fight | teamfight | Team fight is within range (844 units) of Radiant's tier 1 middle tower |
| radiant\_t1\_bottom\_in\_range\_team\_fight | teamfight | Team fight is within range (844 units) of Radiant's tier 1 bottom tower |
| radiant\_t2\_top\_in\_range\_team\_fight | teamfight | Team fight is within range (844 units) of Radiant's tier 2 top tower |
| radiant\_t2\_mid\_in\_range\_team\_fight | teamfight | Team fight is within range (844 units) of Radiant's tier 2 middle tower |
| radiant\_t2\_bottom\_in\_range\_team\_fight | teamfight | Team fight is within range (844 units) of Radiant's tier 2 bottom tower |
| radiant\_t3\_top\_in\_range\_team\_fight | teamfight | Team fight is within range (844 units) of Radiant's tier 3 top tower |
| radiant\_t3\_mid\_in\_range\_team\_fight | teamfight | Team fight is within range (844 units) of Radiant's tier 3 middle tower |
| radiant\_t3\_bottom\_in\_range\_team\_fight | teamfight | Team fight is within range (844 units) of Radiant's tier 3 bottom tower |
| radiant\_t4\_base\_in\_range\_team\_fight | teamfight | Team fight is within range (844 units) of Radiant's tier 4 base tower |
| dire\_t1\_top\_in\_range\_team\_fight | teamfight | Team fight is within range (844 units) of Dire's tier 1 top tower |
| dire\_t1\_mid\_in\_range\_team\_fight | teamfight | Team fight is within range (844 units) of Dire's tier 1 middle tower |
| dire\_t1\_bottom\_in\_range\_team\_fight | teamfight | Team fight is within range (844 units) of Dire's tier 1 bottom tower |
| dire\_t2\_top\_in\_range\_team\_fight | teamfight | Team fight is within range (844 units) of Dire's tier 2 top tower |
| dire\_t2\_mid\_in\_range\_team\_fight | teamfight | Team fight is within range (844 units) of Dire's tier 2 middle tower |
| dire\_t2\_bottom\_in\_range\_team\_fight | teamfight | Team fight is within range (844 units) of Dire's tier 2 bottom tower |
| dire\_t3\_top\_in\_range\_team\_fight | teamfight | Team fight is within range (844 units) of Dire's tier 3 top tower |
| dire\_t3\_mid\_in\_range\_team\_fight | teamfight | Team fight is within range (844 units) of Dire's tier 3 middle tower |
| dire\_t3\_bottom\_in\_range\_team\_fight | teamfight | Team fight is within range (844 units) of Dire's tier 3 bottom tower |
| dire\_t4\_base\_in\_range\_team\_fight | teamfight | Team fight is within range (844 units) of Dire's tier 4 base tower |
| count\_tower\_kill\_by\_dire | game | Total number of towers killed by team Dire so far in the game |
| count\_tower\_kill\_by\_radiant | game | Total number of towers killed by team Radiant so far in the game |
| count\_barraks\_kill\_by\_dire | game | Total number of barracks killed by team Dire so far in the game |
| count\_barraks\_kill\_by\_radiant | game | Total number of barracks killed by team Radiant so far in the game |
| count\_roshan\_kill\_by\_dire | game | Total number of Roshan kills by team Dire so far in the game |
| count\_roshan\_kill\_by\_radiant | game | Total number of Roshan kills by team Radiant so far in the game |


## Dataset Statistics

Approximately 2000 matches were sampled to obtain these statistics. An issue with data processing at the time this code was run reduced the sample size for ability columns when players had more than 5 abilities. For normalization, all ability column statistics were merged to achieve a similar sample size.

## General Game Information

| Metric | Min | Max | Mean | Std |
| :---- | :---- | :---- | :---- | :---- |
| time | 0.0 | 4320.0 | 1464.08 | 943.73 |
| human\_players | 10.0 | 10.0 | 10.0 | 0.0 |
| game\_mode | 1.0 | 22.0 | 11.19 | 9.89 |
| first\_blood\_time | 0.0 | 330.0 | 91.57 | 84.18 |
| radiant\_gold\_adv | \-42645.0 | 39687.0 | 377.93 | 12655.59 |
| radiant\_xp\_adv | \-45126.0 | 40802.0 | 547.90 | 13204.16 |

### Player Statistics (Radiant Team \- Players 0-4)

### Player 0

| Metric | Min | Max | Mean | Std |
| :---- | :---- | :---- | :---- | :---- |
| gold | 0.0 | 32601.0 | 8349.90 | 6814.68 |
| last hits (lh) | 0.0 | 409.0 | 63.86 | 67.17 |
| experience (xp) | 0.0 | 32989.0 | 9568.18 | 8043.50 |
| hero\_id | 2.0 | 110.0 | 53.32 | 37.21 |

### Player 1

| Metric | Min | Max | Mean | Std |
| :---- | :---- | :---- | :---- | :---- |
| gold | 0.0 | 41583.0 | 8633.93 | 7539.75 |
| last hits (lh) | 0.0 | 327.0 | 62.45 | 66.26 |
| experience (xp) | 0.0 | 32776.0 | 9896.87 | 8757.22 |
| hero\_id | 4.0 | 112.0 | 52.68 | 31.72 |

### Player 2

| Metric | Min | Max | Mean | Std |
| :---- | :---- | :---- | :---- | :---- |
| gold | 0.0 | 37690.0 | 8783.46 | 7614.85 |
| last hits (lh) | 0.0 | 348.0 | 65.77 | 64.78 |
| experience (xp) | 0.0 | 32939.0 | 10151.46 | 8971.47 |
| hero\_id | 4.0 | 100.0 | 48.28 | 31.09 |

### Player 3

| Metric | Min | Max | Mean | Std |
| :---- | :---- | :---- | :---- | :---- |
| gold | 0.0 | 31932.0 | 8163.92 | 6663.09 |
| last hits (lh) | 0.0 | 271.0 | 60.94 | 58.70 |
| experience (xp) | 0.0 | 33515.0 | 9549.05 | 8172.49 |
| hero\_id | 1.0 | 112.0 | 45.08 | 34.65 |

### Player 4

| Metric | Min | Max | Mean | Std |
| :---- | :---- | :---- | :---- | :---- |
| gold | 0.0 | 35916.0 | 9063.27 | 7629.02 |
| last hits (lh) | 0.0 | 416.0 | 83.44 | 93.01 |
| experience (xp) | 0.0 | 32642.0 | 10742.86 | 9174.29 |
| hero\_id | 1.0 | 101.0 | 37.64 | 30.90 |

## Player Statistics (Dire Team \- Players 128-132)

### Player 128

| Metric | Min | Max | Mean | Std |
| :---- | :---- | :---- | :---- | :---- |
| gold | 0.0 | 32497.0 | 8555.22 | 7202.37 |
| last hits (lh) | 0.0 | 260.0 | 66.50 | 65.48 |
| experience (xp) | 0.0 | 32646.0 | 9562.17 | 8614.32 |
| hero\_id | 7.0 | 112.0 | 49.44 | 30.61 |

### Player 129

| Metric | Min | Max | Mean | Std |
| :---- | :---- | :---- | :---- | :---- |
| gold | 0.0 | 39682.0 | 9135.10 | 7586.05 |
| last hits (lh) | 0.0 | 584.0 | 73.65 | 77.79 |
| experience (xp) | 0.0 | 32531.0 | 10307.58 | 8852.23 |
| hero\_id | 4.0 | 100.0 | 44.97 | 27.56 |

### Player 130

| Metric | Min | Max | Mean | Std |
| :---- | :---- | :---- | :---- | :---- |
| gold | 0.0 | 38700.0 | 8255.49 | 6988.47 |
| last hits (lh) | 0.0 | 336.0 | 59.52 | 63.88 |
| experience (xp) | 0.0 | 32596.0 | 9374.62 | 8186.00 |
| hero\_id | 5.0 | 105.0 | 45.96 | 30.80 |

### Player 131

| Metric | Min | Max | Mean | Std |
| :---- | :---- | :---- | :---- | :---- |
| gold | 0.0 | 35605.0 | 8900.78 | 7112.78 |
| last hits (lh) | 0.0 | 287.0 | 66.17 | 61.04 |
| experience (xp) | 0.0 | 33102.0 | 10233.53 | 8592.63 |
| hero\_id | 11.0 | 112.0 | 52.27 | 29.62 |

### Player 132

| Metric | Min | Max | Mean | Std |
| :---- | :---- | :---- | :---- | :---- |
| gold | 0.0 | 29994.0 | 8496.75 | 6821.15 |
| last hits (lh) | 0.0 | 278.0 | 64.63 | 60.07 |
| experience (xp) | 0.0 | 32625.0 | 10066.32 | 8543.38 |
| hero\_id | 2.0 | 112.0 | 45.56 | 34.65 |

## Team Fight and Objective Statistics

| Metric | Min | Max | Mean | Std |
| :---- | :---- | :---- | :---- | :---- |
| active\_team\_fight | 1.0 | 1.0 | 0.24 | 0.43 |
| team\_fight\_deaths\_per\_s | 0.0 | 0.25 | 0.02 | 0.04 |
| team\_fight\_damage\_per\_s | 0.0 | 732.77 | 46.14 | 99.30 |
| count\_tower\_kill\_by\_dire | 0.0 | 11.0 | 2.53 | 2.66 |
| count\_tower\_kill\_by\_radiant | 0.0 | 10.0 | 2.23 | 2.50 |
| count\_barraks\_kill\_by\_radiant | 0.0 | 7.0 | 0.64 | 1.21 |
| count\_roshan\_kill\_by\_dire | 0.0 | 4.0 | 0.32 | 0.70 |
| count\_roshan\_kill\_by\_radiant | 0.0 | 2.0 | 0.10 | 0.32 |


## Model Performance





## Citations:
Some scripts in this repo were generated by Claude Sonnet 3.5. These contain the following citation in the header:
```
Citation: Anthropic. (2024). Claude 3.5 Sonnet [Large Language Model]. Retrieved from https://www.anthropic.com
```
All other code was written by Sean Lang-Brown, though some is adapted from reference materials listed in the paper.
