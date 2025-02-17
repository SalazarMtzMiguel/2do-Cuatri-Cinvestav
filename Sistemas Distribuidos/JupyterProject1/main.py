import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from jax import numpy as jnp
import jax
import jax.scipy as jsp
from metaflow import FlowSpec, step
from fastapi import FastAPI
import uvicorn
import threading

app = FastAPI()

class HeartDiseaseFlow(FlowSpec):

    @step
    def start(self):
        """Inicio del flujo de trabajo"""
        self.df = pd.read_csv("heart.csv")
        print("Datos cargados correctamente.")
        sns.pairplot(self.df, hue='HeartDisease')
        plt.show()
        self.next(self.preprocess)

    @step
    def preprocess(self):
        """Preprocesamiento de datos y normalizacion"""
        categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

        self.X = self.df.drop(columns=['HeartDisease'])
        self.y = self.df['HeartDisease']

        for col in categorical_features:
            self.X[col] = self.X[col].astype('category').cat.codes

        self.X = jnp.array(self.X.values)
        self.y = jnp.array(self.y.values)

        self.X_train, self.X_test = self.X[:int(0.8*len(self.X))], self.X[int(0.8*len(self.X)):]
        self.y_train, self.y_test = self.y[:int(0.8*len(self.y))], self.y[int(0.8*len(self.y)):]

        self.next(self.train_models)

    @step
    def train_models(self):
        """Entrenamiento de modelos de clasificacion con JAX"""
        def sigmoid(x):
            return 1 / (1 + jnp.exp(-x))

        def loss_fn(params, X, y):
            logits = jnp.dot(X, params)
            predictions = sigmoid(logits)
            return -jnp.mean(y * jnp.log(predictions) + (1 - y) * jnp.log(1 - predictions))

        self.models = {}
        for model_name in ["Logistic Regression", "Linear Classifier"]:
            params = jnp.zeros(self.X_train.shape[1])
            grad_fn = jax.grad(loss_fn)

            for _ in range(1000):
                grads = grad_fn(params, self.X_train, self.y_train)
                params -= 0.01 * grads

            self.models[model_name] = params

        print("Modelos entrenados correctamente.")
        self.next(self.evaluate)

    @step
    def evaluate(self):
        """Evaluacion de los modelos entrenados"""
        for model_name, params in self.models.items():
            predictions = 1 / (1 + jnp.exp(-jnp.dot(self.X_test, params)))
            predictions = (predictions > 0.5).astype(int)
            accuracy = jnp.mean(predictions == self.y_test)
            print(f"Modelo {model_name} - Precision: {accuracy}")
        self.next(self.api_setup)

    @step
    def api_setup(self):
        """Configuracion de la API con FastAPI"""
        @app.get("/predict")
        def predict(age: float, sex: int, cp: int, trestbps: float, chol: float, fbs: int, restecg: int, thalach: float, exang: int, oldpeak: float, slope: int):
            user_input = jnp.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope])
            prediction = 1 / (1 + jnp.exp(-jnp.dot(user_input, self.models["Logistic Regression"])))
            return {"probabilidad_enfermedad": float(prediction)}

        def run_api():
            uvicorn.run(app, host="0.0.0.0", port=8000)

        api_thread = threading.Thread(target=run_api)
        api_thread.start()
        self.next(self.end)

    @step
    def end(self):
        """Finalizacion del flujo de trabajo"""
        print("Pipeline completado con exito.")

if __name__ == "__main__":
    HeartDiseaseFlow()