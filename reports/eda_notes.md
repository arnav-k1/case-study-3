# EDA Notes — MovieLens 32M (full data)

Source tables: `reports/tables/dataset_summary.csv`, `eda_*.csv`. Figures: `reports/figures/eda_*.png`. Times converted from UTC to US Central.

## Scale and sparsity
- 32,000,204 ratings by 200,948 users on 84,432 rated movies (87,585 in the catalog); 2,000,072 tag applications.
- The user × movie matrix is **99.81% empty** (density 0.19%).
- Mean rating 3.54 stars. 31.3% of ratings use half stars.

## Things observed (surprising or useful)
1. **Most ratings are backfilled at sign-up.** 61.4% of all ratings were entered within 24 hours of the user's first rating, and **58.8% of users entered every rating on their first day** (median user: 100% on day one). Rating timestamps therefore mostly record *when someone joined MovieLens*, not when they watched a film. This matches Harper & Konstan (2015) and shaped the when-to-watch design: only returning-session ratings (38.6% of all ratings) carry timing signal.
2. **Extreme long tail.** The median movie has only **5 ratings**, and 48.0% of movies have fewer than 5. The top 1% of movies receive **54.8%** of all ratings. Only 16,034 movies have ≥ 50 ratings. Popularity is therefore a strong baseline, and a model that ignores it will surface obscure titles.
3. **Heavy users.** The median user has 73 ratings, the mean is 159, and the maximum is 33,332. A handful of users dominate activity.
4. **Tagging is a niche activity.** Only 15,848 users (7.9%) ever tagged, and the top 1% of taggers wrote 65.5% of tag applications. Tags are rich but reflect a small, enthusiastic minority.
5. **Genre quality vs quantity.** Drama (14.0M) and Comedy (11.2M) dominate volume. Film-Noir (3.92), War (3.79), and Documentary (3.69) have the highest mean ratings; Horror has the lowest (3.31). Niche genres are rated by self-selected fans.
6. **Activity waves by year.** Ratings peak in 2000 (1.91M), fall to ~0.5M in 2014, then jump to 1.74–1.92M in 2015–2017, which coincides with the MovieLens redesign period described by Harper & Konstan (2015). Yearly volume reflects site changes, not audience demand.
7. **Weekly rhythm is mild.** Sunday (15.6%) and Monday (15.0%) are the busiest days, Thursday the quietest (12.9%). Activity by hour (US Central) peaks around 15:00, 2.3× the 04:00 trough. Users are worldwide, so the hour of day is not a local clock and is not used for suggestions.
8. **Seasonality exists, but mostly around holidays.** Using returning-session ratings of films at least 2 years old, Horror's share rises to 1.12× in October; Musical, Film-Noir, and Children's films lean toward November–December; Documentary drops to 0.91× in November–December. Christmas-tagged films reach 1.21× in December and Halloween-tagged films 1.22× in October (`wtw_seasonal_lift.png`). Other "seasonal" tags (summer, winter, Valentine's) show no lift.
9. **Users call some films "long".** The tags "too long" (623 users) and "long" (506 users) are among the most-used descriptive tags. This is the only length signal in the data, since runtimes are not included.

## Assumptions carried forward
- Ratings ≥ 4.0 stars indicate a movie the user liked (relevance threshold).
- A rating in a returning session is a better (still imperfect) proxy for recent viewing than a sign-up rating.
- Behavior of MovieLens users (enthusiasts who rated ≥ 20 films) may not represent casual viewers.
