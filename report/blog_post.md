# Is NBA Clutch Performance Real? 26 Seasons of Data Say Yes — With Caveats

---

## Introduction

The NBA has a formal definition of "clutch" that most people don't know about: a game within five points in the final five minutes. Not a fourth-quarter buzzer-beater. Not a playoff elimination game. Any possession in that window — a free throw, a rebound, a turnover — counts. It's a narrower window than the folklore, and that's exactly what makes the data interesting.

For most of basketball's history, clutch was a vibe. Jordan hit the shot. Reggie made the crowd do the cradle. Kobe ball. The analytics era spent a decade poking at whether "clutch" was a measurable skill or a post-hoc story we told about whoever happened to score last. The answer, it turns out, is somewhere specific: clutch performance is real, repeats year over year at a meaningful rate, and partially transfers to the playoffs — but it's noisy enough that every season leaves room for surprises.

The Knicks are a good entry point for why this matters. Over the last two seasons — 2024-25 and 2025-26 — Jalen Brunson has been the most impactful clutch performer in the playoffs among all qualifying players, accumulating 2.37 clutch Win Probability Added over that stretch. His arc tells the full story: in 2018-19, his clutch WPA was -0.51. By 2024-25, his combined regular-season and playoff clutch production hit 4.47 — the highest single-season mark of his career. That's not variance. Players don't casually swing five points on this metric over five years without something real happening.

We looked at 26 seasons of data to understand how predictable that kind of trajectory is, and what box-score statistics can and can't explain about it.

---

## The Data

The clutch metric throughout this analysis is **cWPA — clutch Win Probability Added**, sourced from inpredictable.com. It captures a player's net impact on win probability during clutch situations (within five points, last five minutes), accounting for shots, turnovers, free throws, rebounds, assists, steals, and blocks. Positive cWPA means the player moved the needle in the right direction; negative means the opposite. The methodology is inpredictable.com's own, and while we can describe the inputs, the underlying win probability model isn't publicly auditable.

The dataset covers **26 seasons** (2000-01 through 2025-26) in three variants: regular season, playoffs, and combined. Minimum 62 games for regular-season analysis, minimum 10 for playoffs — thresholds that filter out late-season call-ups, injury-shortened seasons, and first-round sweep victims without meaningful sample sizes. After filtering: roughly 5,500 qualifying regular-season player-seasons and 1,900 playoff entries.

For the predictive model, we paired cWPA with standard **Basketball-Reference box-score stats** — points, shooting efficiency, usage, win shares, and so on. The model asks what a player's regular-season numbers would predict about their clutch production in the same season. It's descriptive, not a crystal ball.

---

## What the Leaderboards Look Like

Before the correlations, it's worth understanding the shape of the data. clutch WPA is right-skewed: most qualifying players sit around 0.2 per season, and genuinely high-impact clutch performers form a thin tail. The median qualifying regular-season cWPA is 0.21. The all-time single-season record is 5.84.

That record belongs to **DeMar DeRozan, 2021-22** — 5.84 cWPA in 75 games, a number that stands apart from everything else in the dataset. The next-closest regular-season season is LeBron James in 2007-08 (5.48). After that, De'Aaron Fox 2022-23 (5.44), Isaiah Thomas 2016-17 (5.19), and Shai Gilgeous-Alexander 2025-26 (4.90) round out the top five. These are generational-pace seasons, not just good years.

The all-time career leaders in regular-season clutch WPA (across all 26 seasons):

| Player | Career cWPA |
|--------|-------------|
| LeBron James | 45.00 |
| Kevin Durant | 35.00 |
| DeMar DeRozan | 32.02 |
| Stephen Curry | 30.05 |
| Dirk Nowitzki | 29.78 |

LeBron leads, which is partly longevity — he appears in more qualifying seasons than almost anyone. The more interesting note is Durant and DeRozan. DeRozan's 32.02 career total includes some early negative seasons; it's his consistency across the last decade that pushed him this high.

Over the last five years (2021-2025), the picture shifts toward the current generation:

**Regular season (G≥62):**
DeRozan leads at 20.55 — by a meaningful margin over Nikola Jokic (13.52) and Shai Gilgeous-Alexander (13.41). Brunson sits fourth at 9.54.

**Playoffs (G≥10):**
Brunson leads at 2.37, with Tyrese Haliburton second at 2.24, Derrick White third (2.14), and Jimmy Butler fourth (2.11).

The best single playoff clutch season in the 26-year dataset: **Haliburton, 2024-25, at 2.63**. Dirk Nowitzki's 2010-11 run (2.15) held that record for over a decade. Haliburton broke it by nearly half a win of probability added in a single postseason.

---

## Do Players Repeat Their Clutch Production?

The central question. We ran three separate analyses — regular season year-over-year, regular season to same-season playoffs, and combined year-over-year — to measure how much clutch production persists.

### Regular Season, Year Over Year

Correlation across 3,244 consecutive season pairs, 25 transitions (2000-01 through 2024-25):

**Pearson r = 0.509. Spearman ρ = 0.492.**

In plain terms: knowing a player's clutch WPA last year explains about 26% of the variance in this year's clutch WPA (r² = 0.26). That's meaningful. It's not luck. For comparison, offensive rebounding rate from year to year shows similar-order persistence — it's a real trait, not perfectly stable, but one that leaves a traceable signal across seasons.

The skeptic's argument — that clutch performance doesn't repeat and we're just making up stories — doesn't survive 3,244 data points and a correlation that holds across every era we tested. The per-transition bar chart shows the r value staying consistently positive from the early 2000s through 2024-25. No single decade is driving the result.

The consistency leaders underscore this. **Chris Paul: 13 qualifying regular seasons, 13 positive cWPA seasons.** Same for **Kevin Durant: 13 for 13.** That's not a streak you run by accident.

The important caveat: r = 0.509 means 74% of the variance is still unexplained. You cannot read last year's cWPA and precisely rank next year's clutch performers. The signal is real but wide. The 95% confidence interval for r is [0.483, 0.534] — directionally right, not a precise forecast.

### Regular Season to Same-Season Playoffs

Correlation across 1,352 player-seasons where a player qualified in both regular season and playoffs in the same year:

**Pearson r = 0.366. Spearman ρ = 0.322.**

Lower than year-over-year, and that gap is the finding. Playoff clutch is harder to predict from the same season's regular-season production than regular-season clutch is to predict from the prior year.

Why the drop? Smaller samples — even a full playoff run has fewer clutch possessions than a regular season. Higher opponent quality. Different game plans, targeted schemes, physical attrition across a series. All of it adds noise. r = 0.37 means 86% of the variance in playoff clutch WPA is unexplained by the regular season. The correlation is real (p effectively zero across 1,352 pairs), but the scatter is wide enough that any individual result can swing dramatically from expectation.

Brunson is a player who consistently beat this relationship — his regular-season clutch numbers, solid but not elite, routinely understated his playoff output. More on that in the next section.

### Combined Year Over Year

Using the combined variant — regular season and playoffs folded together — and running the same year-over-year correlation:

**Pearson r = 0.553. Spearman ρ = 0.525. n = 2,243.**

Higher than regular-season YoY. When you have evidence from both settings, the measurement of the underlying trait improves. A player who is clutch in the regular season and the playoffs in the same year gives you stronger evidence of a real skill than either alone.

The practical implication: combined cWPA is the best available predictor of future clutch performance. If you want to forecast who will be reliable in clutch situations next year, look at their combined number — not just the regular-season line.

The three-prong answer: clutch is real (r ≈ 0.51 year-over-year), transfers to the playoffs at a discount (r ≈ 0.37), and is most stable when you fold in both settings (r ≈ 0.55). Players at the top tend to stay near the top. Not always, and never exactly — but consistently enough to matter.

---

## What Box Scores Can (and Can't) Explain

We trained a machine learning model — XGBoost, which builds an ensemble of decision trees that handle complex, nonlinear interactions between stats — on 21 seasons of data (2000-01 through 2020-21), then tested it on the five held-out seasons it had never seen (2021-22 through 2025-26). The input: 44 standard box-score statistics, narrowed to the 20 most predictive. The output: how much clutch WPA would we expect from this player's regular-season stats?

On those five test seasons, the **regular-season model explains about 46% of the variance** in clutch production (test R² = 0.46). The **playoff model: 27%** (test R² = 0.27). Both are meaningful — better than random, significantly better than just predicting the league average for everyone. But more than half the variance in both cases is unexplained. The gap between the model's prediction and actual output is where clutch separates itself from box-score basketball.

### What the Models Learned

The two models landed on different feature profiles, which itself is informative.

**Regular season** leans on efficiency: **True Shooting % (TS%)** leads, followed by turnover rate, effective field goal %, scoring volume, and offensive box plus/minus (OBPM). The regular-season model is essentially saying: players who convert at a high rate and protect the ball in 82 games tend to produce positive clutch WPA. Efficient scorers bring their efficiency into late-game situations.

**Playoffs** leads with cumulative value: **VORP** (Value Over Replacement Player) tops the list, followed by offensive win shares, effective field goal %, free throws, and win shares. Playoff clutch production tracks more with overall player quality than with any specific efficiency measurement. The best players show up more in the playoffs' clutch moments — not because they're more precise, but because they're simply better players getting more volume.

Both models converge on the same underlying truth: good players tend to be more clutch. The regular model captures the precision component; the playoff model captures the quality component. Neither captures everything.

### The Players Who Beat the Model

The most interesting output is the list of players whose actual clutch WPA far exceeded what their box score would predict. We call this the cWPA Difference (predicted minus actual) — a negative number means the player over-delivered.

**Regular season over-performers, 2021-2025:**

| Season | Player | Predicted | Actual | Difference |
|--------|--------|-----------|--------|------------|
| 2022-23 | De'Aaron Fox | 1.44 | 5.44 | -4.00 |
| 2021-22 | DeMar DeRozan | 2.38 | 5.84 | -3.46 |
| 2024-25 | LeBron James | 1.29 | 3.83 | -2.54 |
| 2022-23 | Dennis Schroder | 0.22 | 2.59 | -2.37 |
| 2022-23 | DeMar DeRozan | 2.32 | 4.63 | -2.31 |

De'Aaron Fox's 2022-23 season is the single biggest regular-season miss in the test data — a gap of four full wins of probability added between what his box score implied and what he actually delivered. DeRozan still appears three times in the top ten over-performers across five test seasons — 2021-22, 2022-23, and 2023-24. His shot selection looks inefficient in the aggregate: he's a mid-range-heavy player in an era that has largely abandoned the mid-range. But he converts those shots in clutch situations at an elite rate, and no box-score metric captures that split. The model sees a mid-range shooter and projects an average clutch output. DeRozan delivers something else.

**Playoff over-performers, 2021-2025:**

| Season | Player | Predicted | Actual | Difference |
|--------|--------|-----------|--------|------------|
| 2024-25 | Tyrese Haliburton | 0.29 | 2.63 | -2.34 |
| 2022-23 | Jamal Murray | -0.09 | 1.23 | -1.32 |
| 2024-25 | Aaron Gordon | 0.22 | 1.30 | -1.08 |
| 2022-23 | Derrick White | 0.34 | 1.41 | -1.07 |
| 2022-23 | Jimmy Butler | 0.74 | 1.77 | -1.03 |

Haliburton's 2024-25 playoff season is the model's biggest miss in the entire playoff dataset. The model projected 0.29 cWPA — a modest, above-average output consistent with his regular-season numbers and his prior playoff performance (he was actually a model under-performer in 2023-24, posting -0.39 when the model expected 0.31). In 2024-25, he delivered 2.63 — the best single playoff clutch season in 26 years. The model had no mechanism to predict that.

### The Players Who Disappointed the Model

The under-performers are equally revealing.

**Regular season under-performers, 2021-2025:**

| Season | Player | Predicted | Actual | Difference |
|--------|--------|-----------|--------|------------|
| 2021-22 | Donovan Mitchell | 1.63 | -0.84 | +2.47 |
| 2021-22 | Trae Young | 2.49 | 0.03 | +2.46 |
| 2022-23 | Jayson Tatum | 2.57 | 0.29 | +2.28 |

Donovan Mitchell and Trae Young share the two biggest regular-season misses in the test data, separated by 0.006 cWPA — effectively the same outcome. Mitchell's efficiency stats implied a strong clutch producer; the model expected 1.63 cWPA, he posted -0.84. Young's efficiency was similarly elite — TS%, OBPM, scoring volume all pointing toward strong clutch production, a 2.49 projection, and 0.03 in reality. Neither was hurt; both were not delivering in the specific five-minutes-within-five-points window that season, whatever else they were doing in a box score.

Jayson Tatum appears twice (2022-23 and 2023-24) as a significant under-performer. His regular-season production and efficiency suggest elite clutch output; he's repeatedly come in below that expectation.

### What This Means

The model's failures point at something the stat sheet can't measure in aggregate — shot selection within clutch situations, how a player's tendencies shift under pressure, or simply the random variance that comes with small samples. DeRozan, Brunson, and Haliburton at his 2024-25 peak delivered something that efficient regular-season production didn't predict. Trae Young and Tatum repeatedly fell short of what their production would imply.

Whether that gap is a stable, learnable skill or a combination of tendencies and noise is impossible to disentangle with this data. But the over-performers list is a reasonable shortlist of players you'd want taking the ball with the game on the line — not because they'll always deliver, but because they've consistently delivered when the model gave them no particular reason to.

---

## Conclusion

Three findings, stated plainly.

Clutch performance **repeats** from year to year at a meaningful rate (r ≈ 0.51). It is a real, measurable trait — not noise, not narrative. The players at the top of the leaderboard in one season are more likely than not to be above average the next. Chris Paul was positive in all 13 of his qualifying regular seasons. Kevin Durant, same. That consistency doesn't happen by accident.

Clutch performance **transfers to the playoffs at a discount** (r ≈ 0.37). The signal is real but the noise is larger — smaller samples, better opponents, physical wear. 86% of the variance in playoff clutch output goes unexplained by the regular season. The correlation matters enough to inform expectations; it doesn't determine them.

**Combined production** — both regular season and playoffs — is the best predictor of future clutch ability (r ≈ 0.55). If you want to know who will be reliably impactful in high-stakes situations next year, look at the combined ledger, not just regular-season totals.

Brunson's trajectory from -0.51 combined in 2018-19 to 4.47 in 2024-25 is a genuine outlier in this dataset — a player who became dramatically more clutch as he matured. The data would have caught it early. The regular-season signal was building from 2022-23 onward. What it couldn't predict was the ceiling.

Haliburton's 2024-25 postseason — 2.63 cWPA, the best single playoff clutch performance in 26 years — is a reminder that the model is not the ceiling of what's possible. It describes what tends to happen. It cannot describe Haliburton putting up the best clutch playoff season since Dirk won a championship, from a box-score profile that gave no indication he was about to.

That's probably the honest conclusion of this whole project. Clutch is real. It's not deterministic. The players who do it consistently over many years are doing something that box scores partially explain and largely cannot. That's not a gap in the analysis. That's just how hard basketball is to reduce to a spreadsheet.

---

*Data: inpredictable.com (cWPA, 2000-01 through 2025-26), Basketball-Reference (box-score stats). Analysis covers all qualifying regular-season (G≥62) and playoff (G≥10) player-seasons.*
