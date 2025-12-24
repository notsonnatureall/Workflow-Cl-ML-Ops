import pandas as pd
import mlflow
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
args = parser.parse_args()

def run():
    df = pd.read_csv('financial_regression_preprocessing.csv')
    X = df.drop(['gold high', 'Unnamed: 0'], axis=1)
    y = df['gold high']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    with mlflow.start_run(run_name="Basic_CI_Trigger"):
        model = RandomForestRegressor(n_estimators=args.n_estimators)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        print("Training sukses!")

if __name__ == "__main__":
    run()