# The trajectory model, in plain English

This note explains the mixed-effects model in `src/trajectory.py`: what it does, what its numbers mean, how far to trust it, and where it bends the truth. The numbers come from the files named next to them in `data/processed/` (built by running the pipeline; not in git). They are from the run on 23 Nov 2026, after the survivorship fix.

## 1. The question

How much Premier League football does a player get as he gets older, and what does that suggest for a young player's future? The outcome is **minutes share**: PL minutes in a season divided by 3,420 (38 matches of 90 minutes). 0 means he did not play in the PL that season; 1 means he played every minute.

## 2. What the model sees, and why only players first seen young

One row per player per season (`player_season_panel.parquet`), at ages 17 to 30. Age is measured on 1 September at the start of the season.

A season where a player has dropped out of the league (to the Championship, abroad, or out of the game) is kept as a 0, not left out. If those seasons were dropped, the model would only see players who stayed in the league, and it would think young players do better than they do.

**The data starts in 2016-17, and that creates a trap (left truncation).** A player who was already 25 in 2016-17 is only in the data because he was still in the league at 25. Everyone his age who had already dropped out is invisible. So among players first seen in their twenties, the data only has survivors, and survivors play a lot. The first version of the model learnt from all players, and it expected far too much of young players at 23 to 26: in a test on past seasons it over-predicted by 16 percentage points on average.

The fix: **the model is now fitted only on players first seen young**, meaning their first PL season with minutes came at 20 or under (the same ages as the under-21s it projects). For those players the data sees the whole story from their debut, including any drop-out. That leaves 2,680 seasons from 515 players, against 7,322 from 1,767 before (`trajectory_model_summary.txt`). About half of these seasons (51%) are zeros, and 996 of the 2,680 are seasons after a player left the league, filled in as 0.

This is not a perfect fix. Some players first seen at 21 or older in our data had in fact played in the PL younger, before 2016-17. A check against Transfermarkt finds at least 75 of them, and they are left out (`truncation_check.csv`).

## 3. The equation

For player *i* in a season when he is *age* years old, with `age_c = age - 21`:

```
minutes share = b0 + b1 x age_c + b2 x age_c^2 + position shift     (the average curve: fixed effects)
              + u0_i + u1_i x age_c                                  (this player's own deviation: random effects)
              + noise                                                (season-to-season ups and downs)
```

### Fixed effects: the average curve

| term | estimate | what it means |
| --- | --- | --- |
| `b0` (Intercept) | 0.180 | An average 21-year-old defender plays about 18% of PL minutes. Age is centred at 21, so the intercept is "at 21". |
| `b1` (age_c) | +0.017 | Around 21, minutes share is still rising, by under 2 percentage points a year. |
| `b2` (age_c squared) | -0.005 | The rise slows each year and turns into a decline, so the curve is an upside-down U. |
| position shifts | FWD -0.080, MID -0.022, GK +0.012 | Compared with defenders at the same age. Young forwards get the fewest minutes. |

The average curve peaks where its slope is zero: `21 - b1 / (2 x b2)`, which is **age 22.8**. Position only moves the curve up or down, so the peak age is the same for every position. At its peak, the average defender who broke through young plays about 19% of minutes. The first version of the model, fitted on all players, said 35% at 24, because it learnt the late twenties from survivors.

### Random effects: one player's own curve

Every player gets two personal adjustments:

- **Random intercept `u0`**: how far above or below the average curve he sits at age 21.
- **Random slope `u1`**: how much faster or slower his minutes change with age.

Across players, these adjustments are assumed to follow a normal distribution with mean zero. The model estimates how spread out they are:

| quantity | estimate | in words |
| --- | --- | --- |
| SD of `u0` | 0.161 | At 21, players differ a lot: about two thirds sit within 16 percentage points of the average curve. |
| SD of `u1` | 0.035 per year | Some players' minutes grow clearly faster than others'. |
| correlation of `u0` and `u1` | 0.44 | Players who are ahead at 21 also tend to be rising faster. |
| SD of the noise | 0.156 | A single season is noisy: injuries, a change of manager, a loan. |

About half the variation at age 21 (an intraclass correlation of 0.51) is between players; the rest is season-to-season noise. Allowing a random slope improves the fit clearly (likelihood ratio statistic 237, against a model with a random intercept only).

## 4. Shrinkage: why one good season is not taken at face value

The model never takes a player's record at face value. It **shrinks** the gap between his seasons and the average curve towards zero (towards the average player), and how much depends on how much evidence there is.

A simplified version with only a random intercept makes the idea concrete. The weight given to a player's own average gap is

```
weight = between-player variance / (between-player variance + noise variance / number of seasons)
```

With the fitted values (0.0258 between players, 0.0245 noise):

| seasons of data | weight on his own record | weight on the average |
| --- | --- | --- |
| 1 | 0.51 | 0.49 |
| 2 | 0.68 | 0.32 |
| 5 | 0.84 | 0.16 |

So for a 19-year-old with **one** season of 2,000 minutes (share 0.58), the model believes about half of the gap between him and the average 19-year-old. With two strong seasons it believes more. This stops the model from crowning a teenager after one bright half-season, or writing off a player after one injury-hit year.

**The random slope barely moves projections for players with one or two seasons.** A slope needs several ages to be seen, so with little history it is shrunk almost to zero. For the 2025/26 under-21s, the slope moves the projected peak by a median of 0.006 (under 1 percentage point) for the 42 players with one season in the model's ages, and 0.013 for the 29 with two. The player's own intercept moves it by 0.048 and 0.061. For the six players with four seasons, the slope's median effect is 0.070 (`pa_projections_2025_26.parquet`, columns `slope_contribution` and `intercept_contribution`). There is one exception: because intercept and slope are correlated, a player with a very large intercept is also given a steeper slope even after one season (up to 0.138 here). In the backtest fit for 2020-21, where the correlation was 0.85, this produced impossible forecasts above 100% (section 7).

The formulas used (`player_posterior` in `src/trajectory.py`) are the standard best linear unbiased predictions (BLUPs). They reproduce statsmodels' own random effects to within 1e-15.

## 5. From the model to a projection, and how the range is widened

For a 2025/26 under-21, the model projects his minutes share at ages 24, 25 and 26 and averages them (the "projected peak"). It reports a middle value and a range, never a single number. The range comes from 2,000 simulated futures, each drawing:

1. the average curve, allowing for uncertainty in its estimates;
2. the player's own adjustments, given his seasons so far (the shrunk estimate plus its uncertainty);
3. fresh season-to-season noise for each of the three seasons.

**Is the range wide enough? (calibration), for a coach.** The range is meant to catch the real outcome 4 times in 5 (an "80% range"). When we tested the model on past seasons (section 7), it caught it only about 2 times in 3 (66%) overall, so we worked out how much the ranges would need stretching to catch it 4 times in 5: **each side 1.43 times as long** (`interval_calibration.json`). Each test season was only ever widened using errors from earlier test seasons, never its own.

**The player cards show the unstretched range.** We decided this after seeing the test results, so it is a judgement call, not a rule fixed in advance. Most of the shortfall came from the early test seasons, when the data had almost no players aged 23 to 25 who had broken through young. There the model was guessing at that age, which is not the case for today's projections. In the one test season that does look like today (2020-21), the unstretched range already caught the real outcome 81% of the time, and stretching pushed it to 92%. The stretched version is still kept in the data as a check.

Even unstretched, the range is honest but wide. For most under-21s it starts at zero, which the card puts in words: "Range: from not playing regularly in the PL to about 45% of league minutes at 24 to 26". Dropping out of the league is a real outcome for young players, and the range says so. Ranges are cut at 0% and 100% for display only. A player whose PL seasons so far all came before age 17 gets **no projection** ("insufficient data: debut before 17"), because the model is not fitted below 17 and could only give him the average curve.

**The 1 to 20 number on the card ("rank among 2025/26 U21 prospects").** The card's PA ranks each player's projected middle value against every other under-21 the model projected this season (85 players, all positions together), in the same way as the attributes: the middle-ranked prospect gets 11 and the top four get 20. It is a ranking of projections against projections. An earlier version compared projections with what past players actually went on to play. That did not work, because a projection is an average that is almost never exactly zero, while half of past young debutants played no PL football at 24 to 26. Any projection above zero then outranked all of them. The card gives no 1 to 20 range; the minutes range in words carries the uncertainty. The number also depends on position: the model expects fewer minutes for young forwards, so forwards rank lower on average.

**Low evidence.** A player with fewer than 450 PL minutes so far is projected almost entirely from the average curve, because his own record says little (section 4). His PA is shown greyed out on the card and in the charts. This season that applies to 49 of the 85 projected players (`pa_projections_2025_26.parquet`, column `pa_low_evidence`).

## 6. Assumptions, and where they bend

- **Normal errors on a bounded outcome.** Minutes share cannot go below 0 or above 1, and half the fitting seasons are exactly 0. A normal (Gaussian) model ignores both facts, so it can predict impossible values: the average curve for a 17-year-old forward is -0.040, and in the 2020-21 backtest three forecasts were above 1. A two-part (hurdle) model or a beta regression would respect the bounds; this is a deliberate first approximation.
- **A quadratic shape.** The curve is an upside-down U that is symmetric around the peak. Real careers may rise faster than they fall. The curve also goes below 0 by age 30, where there is almost no data on players first seen young.
- **Position only shifts the curve.** Every position is assumed to peak at the same age.
- **Seasons are independent once a player's curve is known.** In reality, a bad season (injury, loan) makes the next one more likely to be bad too. This is one reason the raw ranges were too narrow.
- **Personal adjustments are normal.** A few players (regular starters from 18) are far from average; a normal distribution treats them as rare.
- **Who is missing.** Players who never played in the PL are not in the data (survivorship). Seasons after a debut are filled with zeros up to age 27 only, so the few seasons at 28 and 29 come from players still in the league. Some players who debuted young before 2016-17 are wrongly counted as "first seen older" and left out (section 2).
- **Minutes are not ability.** Minutes depend on the club, the manager, relegation and the competition for places. A player at a top club may get fewer minutes than a weaker player at a weaker club.

## 7. How good is it? The backtest

`src/validate.py` refits the model using only seasons up to an earlier date (2017-18 to 2020-21), predicts what that season's under-21s went on to play at ages 23 to 25, and compares the result with what actually happened and with two simple baselines. The baselines are "the minutes share he has had so far" (persistence) and the average curve. Full results are in `backtest_summary.md`.

- Fitting only on players first seen young cut the over-prediction from +0.159 to +0.070 and lowered the average error (MAE) at every test season, from 0.280 to 0.244 over all four.
- **It still does not beat the simple "minutes so far" baseline on average error** (0.244 against 0.214). It ranks players about as well as that baseline: Spearman 0.287 against 0.314 over all four seasons, and 0.419 against 0.375 in the one season where the model had enough data at the target ages.
- That caveat matters: only 2020-21 had enough seasons at ages 23 to 25 among players first seen young (78; the threshold is 50). In the earlier test seasons the curve at those ages was pure extrapolation.

So the projection is labelled **experimental** on the player card, and the card also shows the player's minutes share so far.
