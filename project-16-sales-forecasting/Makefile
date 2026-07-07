.PHONY: setup data train evaluate app clean

setup:
	pip install -r requirements.txt

data:
	python -c "from src.utils import load_config; from src.data_generator import generate_sales_data; c=load_config(); df=generate_sales_data(c); df.to_csv(c['paths']['data'], index=False); print('Synthetic dataset written to', c['paths']['data'])"

train:
	python train.py --model_type both

evaluate:
	python -m src.evaluator

app:
	streamlit run app/streamlit_app.py --server.port 8501

clean:
	rm -rf models/*.joblib reports/figures/*.png reports/*.txt
