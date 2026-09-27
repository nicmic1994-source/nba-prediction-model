# Recommended modelling strategy

## 1. Target

Primary target: home-team win probability.

Secondary target: margin of victory. A joint win/margin model is useful because margin contains more information than a binary result and can improve probability estimates.

## 2. Feature blocks

### Team strength
- TS%
- eFG%
- TOV%
- ORB%, DRB%, REB%
- FT rate
- offensive/defensive/net rating
- pace
- points per possession proxies
- opponent-adjusted versions

Use 5/10/20-game rolling windows, exponentially weighted form, and season-to-date estimates. All are shifted by one game before use.

### Context
- home court
- rest days
- back-to-back
- 3 games in 4 nights
- 4 games in 6 nights
- travel distance and time-zone change
- altitude
- regular-season vs playoffs
- days since last head-to-head

### Player availability
For every player expected to be unavailable/questionable, estimate lost team value as:

`expected_minutes × player_impact × availability_probability`

This is substantially better than simply counting injured players.

For modern seasons, construct an internal player-impact rating from play-by-play lineups/on-off data plus regularized box-score information. For older seasons where detailed stint data is unavailable, use a more conservative box-score value model.

## 3. Era handling

Do not compare a raw 1990s TS% directly with a modern TS% and assume the relationship is unchanged. Normalize many rate statistics within season (z-score or percentile) and include season/era effects. Retain all-time data as a prior, but apply recency weighting for current predictions.

A practical initial scheme is an exponential game-weight with a 3–5 season half-life for the team-strength prior, plus a much faster form component over the current season.

## 4. Model stack

Use three complementary models:

1. Elastic-net logistic regression: transparent weights and stable extrapolation.
2. Gradient boosting: captures nonlinear interactions.
3. Latent strength model (Elo/state-space): captures changing team quality and schedule strength.

Blend their probabilities on validation data. Calibrate the final blend using an out-of-sample calibration set.

## 5. Optimisation objective

Use chronological validation and optimise primarily for:

- log loss
- Brier score
- calibration error
- ROC AUC as a secondary discrimination metric

Accuracy should be reported but should not be the main optimisation target, because a model can have high accuracy while producing poorly calibrated probabilities.

## 6. Confidence / uncertainty

Show four separate concepts:

- **Win probability:** the calibrated probability of the selected outcome.
- **Prediction interval:** e.g. 90% interval from bootstrap/ensemble predictions.
- **Model agreement:** how much component models disagree.
- **Calibration reliability:** how often historical predictions in the same probability band were actually correct.

Example dashboard presentation:

`Boston 67% win probability`  
`90% model interval: 61–72%`  
`Model spread: 11 percentage points`  
`Calibration: well-supported at this probability range`

Do not label 67% probability as 67% certainty. Those answer different questions.

## 7. Walk-forward testing

Recommended production evaluation:

- Train on 1946 through a cutoff.
- Predict the next block.
- Move the cutoff forward.
- Repeat through the latest completed season.

Also run separate reports for:

- full historical period
- last 15 seasons
- last 5 seasons
- current-season games only
- games with known injury/availability information

That reveals whether improvements are genuine or only historical-era effects.

## 8. Leakage traps to prohibit

Never use:

- end-of-season standings to predict earlier games
- a player's post-game stat line for that game
- final injury status when testing a prediction claimed to be made the previous day
- future roster information
- closing betting lines as model features unless the stated prediction time is after the line was available
- future games when calculating rolling averages

Every prediction row should have a `prediction_timestamp` and every input field should have an `as_of_timestamp`.
