# Personal Finance Tracker Dashboard

An end-to-end personal finance analysis project built from `personal_finance_tracker_dataset.csv`.
The project analyzes spending patterns by category, segments users by savings behavior, and turns the findings into a self-contained HTML dashboard.

## Live Dashboard

If GitHub Pages is enabled for this repository, the dashboard is available at:

https://lisslatte.github.io/personal-finance-tracker/

## Project Contents

- `index.html` - GitHub Pages entry point, copied from the generated dashboard.
- `finance_story_dashboard.html` - Self-contained Plotly dashboard with the full story.
- `personal_finance_analysis.ipynb` - Executed Jupyter notebook with analysis, charts, segment explanations, and bank/advisor recommendations.
- `build_finance_analysis.py` - Reproducible Python script that rebuilds the notebook, dashboard, and output CSV files.
- `personal_finance_tracker_dataset.csv` - Source dataset.
- `analysis_outputs/` - Generated enriched data and summary tables.

## Analysis Summary

The analysis includes:

- Category-level spending pattern analysis.
- Savings behavior segmentation using KMeans clustering.
- Segment standards and interpretation for:
  - Resilient investors
  - Goal-focused savers
  - Debt-pressure spenders
  - Cash-flow stretched
- Recommendations for what banks and personal finance advisors can do for each segment.

## Key Findings

- Average savings rate is about 22.6%, but the savings goal-met rate is only about 9.2%.
- Groceries has the highest average monthly expense and the weakest goal-met rate.
- Investments has the strongest category-level goal performance.
- Debt-pressure spenders are the largest segment, covering about 43.5% of users.
- Goal-focused savers are the smallest segment but have the strongest goal achievement.

## Rebuild the Analysis

From this folder, run:

```powershell
python build_finance_analysis.py
```

This regenerates:

- `personal_finance_analysis.ipynb`
- `finance_story_dashboard.html`
- `analysis_outputs/*.csv`

After rebuilding, update the GitHub Pages entry file:

```powershell
Copy-Item "finance_story_dashboard.html" "index.html" -Force
```

Then commit and push:

```powershell
git add .
git commit -m "Update finance dashboard"
git push
```

## GitHub Pages Deployment

In the GitHub repository:

1. Go to `Settings`.
2. Open `Pages`.
3. Set source to `Deploy from a branch`.
4. Select branch `main`.
5. Select folder `/root`.
6. Save.

The dashboard should publish at:

```text
https://lisslatte.github.io/personal-finance-tracker/
```
