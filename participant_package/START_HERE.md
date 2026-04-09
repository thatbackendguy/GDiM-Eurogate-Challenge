# Start Here

Recommended order for participants:

1. Read `REEFER_PEAK_LOAD_CHALLENGE.md`
2. Read `REEFER_DATA_COLUMNS.md` to understand the reefer dataset fields
3. Check `EVALUATION_AND_WINNER_SELECTION.md`
4. Open `starter/reefer_starter_notebook.ipynb` for a first valid submission
5. Use `target_timestamps.csv` as the released public prediction target list
6. Submit a file matching `templates/submission_template.csv`
7. Make sure your code or notebook can be rerun easily by organizers on the hidden full target list and the complete reefer release data

Quick domain translation:

- `reefer container` = refrigerated shipping container
- this challenge predicts combined reefer electricity demand, not general terminal workload
- the main business question is which hours tomorrow are likely to create high reefer power demand
