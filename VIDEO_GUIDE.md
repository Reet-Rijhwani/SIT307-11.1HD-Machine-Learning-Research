# Five-minute presentation outline — personalize and narrate yourself
0:00–0:35: Face visible. Introduce source, prediction task, 13 input features + target.
0:35–1:20: Summarise six methods, 5-fold stacking and reported 98.53% accuracy, highlighting unspecified split/model details.
1:20–2:05: Show heart.csv, duplicates audit (723 duplicates; only 302 unique rows) and train/test duplicate overlap.
2:05–3:05: Screen-share `run_experiments.py`; demonstrate execution and show published-vs-random-split CSV and plot. Explain no test data used for fitted scalers or meta-learner.
3:05–4:15: Show redesigned duplicate-aware unique-row experiments, three-seed summary, ROC and confusion matrix; compare accurately without claiming improvement in predictive accuracy.
4:15–5:00: Describe remaining limits: no individual identifiers, only 302 distinct vectors, small single dataset, different test populations, untuned assumptions, clinical validation needed. Show code/data/README and explain what you independently verified.
Add accessible unlisted video URL to your own report. Video must show face, screen, voice and selected code execution.
