.PHONY: setup data train app clean

setup:
	pip install -r requirements.txt

data:
	python -c "import yaml; from src.data_generator import generate_energy_data; \
config = yaml.safe_load(open('config.yaml')); df = generate_energy_data(config); \
df.to_csv(config['paths']['data'], index=False); print('Data generated and saved.')"

train:
	python train.py

app:
	streamlit run app/streamlit_app.py --server.port 8501

clean:
	rm -rf models/*.joblib models/*.json reports/figures/*.png reports/*.txt
