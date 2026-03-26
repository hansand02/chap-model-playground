prediction:
	chap evaluate --model-name ./ --dataset-name ISIMIP_dengue_harmonized --dataset-country brazil --report-filename new-report.pdf --prediction_length 40

local-prediction:
	uv run train.py input/chap_LAO_admin1_monthly.csv output/model.pkl  --provider ensemble