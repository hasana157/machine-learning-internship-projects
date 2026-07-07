.PHONY: setup train evaluate predict mlflow-ui clean

setup:
	pip install -r requirements.txt

train:
	python train.py --config config.yaml

evaluate:
	python evaluate.py --config config.yaml

predict:
	python predict.py --input sample_applicant.json

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:
	rm -rf artifacts/*.joblib artifacts/*.json artifacts/*.csv reports/figures/*.png mlflow.db
