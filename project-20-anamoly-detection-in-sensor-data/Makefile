.PHONY: setup data train evaluate app clean

setup:
	pip install -r requirements.txt

data:
	python -c "from src.data_generator import generate_sensor_data; import yaml; config=yaml.safe_load(open('config.yaml')); df, _ = generate_sensor_data(config); df.to_csv(config['paths']['data'], index=False); print('Data generated and saved.')"

train:
	python train.py

evaluate:
	python src/evaluator.py

app:
	streamlit run app/streamlit_app.py --server.port 8501

clean:
	rm -rf models/*.joblib data/*.csv reports/figures/*.png
