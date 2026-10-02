# Literature and Existing-System Review

This review summarizes how established recommender systems work and states what this project borrows from each. Bibliographic details were checked against the ACM Digital Library, IEEE/dblp, Springer, and publisher pages on 2026-09-26. Direct statistics quoted from Gomez-Uribe and Hunt (2015) were checked against the paper's full text.

## 1. MovieLens and GroupLens

MovieLens is a non-commercial movie recommender operated by the GroupLens research group at the University of Minnesota since 1997. Harper and Konstan (2015) document the history of the site and of the public datasets built from it. They show how interface changes shape the data: half-star ratings began on February 18, 2003, and version 3 of the site required 15 ratings at sign-up. They also cover user-selection rules and later site redesigns. Three consequences matter here:

- Every user in the datasets has at least 20 ratings. Harper and Konstan note that this biases the data toward "successful" users. The dataset contains no truly new users, so cold start has to be simulated.
- New users rate movies during sign-up, and the movies shown to them are chosen by popularity. The first ratings a user enters therefore overrepresent well-known titles.
- Harper and Konstan state that the timestamps "do not represent the date of consumption," and that users often enter many ratings in a single session, backfilling their history. The EDA in this project quantifies this: 58.8% of ML-32M users entered all of their ratings within 24 hours of their first one.

**Borrowed:** the dataset, the pick-movies-you-know onboarding pattern for cold-start users, and caution about reading timestamps as viewing times.

## 2. LensKit

LensKit for Python (Ekstrand, 2020) is a research toolkit for training, running, and evaluating recommender algorithms. It provides reference implementations of classic collaborative-filtering algorithms (bias baselines, user- and item-based k-NN, explicit and implicit ALS matrix factorization) and batch evaluation utilities. Ekstrand argues for reproducible, inspectable experiments rather than ad-hoc scripts. The 2025 releases, used here (version 2025.8.1), reorganized the toolkit around pipelines of components (history lookup → candidate selector → scorer → ranker) and a `Dataset` abstraction.

**Borrowed:** all model implementations, the pipeline structure (including automatic exclusion of already-rated movies), and LensKit's ranking metrics (NDCG, Recall, Precision, reciprocal rank).

## 3. Neighborhood methods and Amazon item-to-item

Sarwar et al. (2001) showed that item-based collaborative filtering, which computes similarities between items rather than users, is both more accurate and more scalable than user-based filtering, because item-item similarities are more stable and can be precomputed. Linden, Smith, and York (2003) describe Amazon.com's production item-to-item system. It builds a similar-items table offline, and online it only looks up neighbors of the items a customer has bought or rated, so response time is independent of catalog and customer counts.

**Borrowed:** item k-NN as a candidate model and, more importantly, as the source of the "Because you liked X" explanations. Its instant scoring of a brand-new user from a handful of ratings is the Amazon property the app's onboarding relies on.

## 4. Matrix factorization (the Netflix Prize lineage)

Koren, Bell, and Volinsky (2009) summarize the latent-factor models that dominated the Netflix Prize. Users and items are mapped to a shared low-dimensional space, rating prediction is an inner product plus user and item biases, and the models are fit by stochastic gradient descent or alternating least squares (ALS). Biases alone explain much of the rating variance, which is why a bias model is a strong baseline. Hu, Koren, and Volinsky (2008) adapt ALS to implicit feedback: every observed interaction becomes a binary preference with a confidence weight, and unobserved cells are treated as weak negatives. That formulation suits top-N ranking better than rating prediction.

**Borrowed:** the bias baseline, explicit biased MF (ALS), and implicit-feedback MF (ALS) as candidate models.

## 5. Netflix

Gomez-Uribe and Hunt (2015) describe the Netflix recommender as a collection of algorithms (personalized video ranker, top-N ranker, trending now, continue watching, video-video similarity, page generation) that together fill a personalized homepage of rows. They report that the recommender influences choice for about 80% of hours streamed. They also report that the combined effect of personalization and recommendations saves Netflix more than $1B per year through reduced churn. Their consumer research found that a typical member loses interest after 60 to 90 seconds of choosing, having reviewed 10 to 20 titles. They also stress that offline metrics only predict online success imperfectly and that A/B tests are the final arbiter.

**Borrowed:** the design target of getting a new user to good recommendations in under one minute, the row/card presentation with explanations ("Because you watched ..."), and the caveat that offline gains need online confirmation.

## 6. YouTube

Covington, Adams, and Sargin (2016) split YouTube's recommender into a deep candidate-generation network, which narrows millions of videos to hundreds, and a separate deep ranking network. They note that "example age" (freshness) and the time a request is made matter to what users want, and they train on watch sessions rather than explicit ratings.

**Borrowed:** the two-stage idea in lightweight form: a collaborative-filtering model generates and scores candidates, and a context stage (filters, hidden-gems re-weighting, when-to-watch) re-ranks them. The deep architecture itself was considered and not adopted (see the report's Methods).

## 7. Context-aware recommendation (the "when" question)

Adomavicius and Tuzhilin (2011) define context-aware recommender systems (CARS). These extend the user × item model with contextual dimensions such as time, location, and companions, and use three paradigms: *contextual pre-filtering* (select data relevant to the context, then recommend), *contextual post-filtering* (recommend, then adjust the list for context), and *contextual modeling* (context inside the model). Baltrunas and Amatriain (2009) study time-dependent recommendation from implicit feedback by splitting a user's profile into time-of-day or day-of-week "micro-profiles."

**Borrowed:** the when-to-watch module is a contextual post-filter. It keeps the collaborative-filtering ranking and attaches a time context (month, weekend or weeknight, holiday window), with an evidence score derived from time-stamped activity.

## 8. Consumer movie apps: Letterboxd and IMDb

Letterboxd, a social film-logging service, does not present a single "recommended for you" feed. Its help center states that each film page shows similar films generated by a Nanocrowd algorithm that compares viewer reviews, along with "nanogenre" themes. Users are also encouraged to use sorting and filtering (for example, films not yet watched, filtered by genre or decade) to find titles (Letterboxd, n.d.). IMDb shows "More like this" lists on title pages. Both apps emphasize browsing controls (genre, decade, popularity) alongside similarity lists.

**Borrowed:** the filter bar (genre, decade, hide already-seen, a popularity or "hidden gems" control) and similar-title explanations.

## 9. Evaluation

Normalized discounted cumulative gain (NDCG) was introduced by Järvelin and Kekäläinen (2002) for graded-relevance retrieval. It rewards placing relevant items near the top of a ranked list. It is the primary top-N metric in this project, alongside recall, precision, and reciprocal rank.

## References (APA 7)

Adomavicius, G., & Tuzhilin, A. (2011). Context-aware recommender systems. In F. Ricci, L. Rokach, B. Shapira, & P. B. Kantor (Eds.), *Recommender systems handbook* (pp. 217–253). Springer. https://doi.org/10.1007/978-0-387-85820-3_7

Baltrunas, L., & Amatriain, X. (2009). Towards time-dependant recommendation based on implicit feedback. In *Proceedings of the Workshop on Context-Aware Recommender Systems (CARS 2009)*. [VERIFY: workshop proceedings venue/URL]

Covington, P., Adams, J., & Sargin, E. (2016). Deep neural networks for YouTube recommendations. In *Proceedings of the 10th ACM Conference on Recommender Systems* (pp. 191–198). ACM. https://doi.org/10.1145/2959100.2959190

Ekstrand, M. D. (2020). LensKit for Python: Next-generation software for recommender systems experiments. In *Proceedings of the 29th ACM International Conference on Information & Knowledge Management* (pp. 2999–3006). ACM. https://doi.org/10.1145/3340531.3412778

Gomez-Uribe, C. A., & Hunt, N. (2015). The Netflix recommender system: Algorithms, business value, and innovation. *ACM Transactions on Management Information Systems, 6*(4), Article 13. https://doi.org/10.1145/2843948

Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context. *ACM Transactions on Interactive Intelligent Systems, 5*(4), Article 19. https://doi.org/10.1145/2827872

Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets. In *2008 Eighth IEEE International Conference on Data Mining* (pp. 263–272). IEEE. https://doi.org/10.1109/ICDM.2008.22

Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. *ACM Transactions on Information Systems, 20*(4), 422–446. https://doi.org/10.1145/582415.582418

Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer, 42*(8), 30–37. https://doi.org/10.1109/MC.2009.263

Letterboxd. (n.d.). *Can Letterboxd generate recommendations for me?* Letterboxd Help Center. Retrieved September 26, 2026, from https://letterboxd.zendesk.com/hc/en-us/articles/15178828078223-Can-Letterboxd-generate-recommendations-for-me

Linden, G., Smith, B., & York, J. (2003). Amazon.com recommendations: Item-to-item collaborative filtering. *IEEE Internet Computing, 7*(1), 76–80. https://doi.org/10.1109/MIC.2003.1167344

Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based collaborative filtering recommendation algorithms. In *Proceedings of the 10th International Conference on World Wide Web* (pp. 285–295). ACM. https://doi.org/10.1145/371920.372071
