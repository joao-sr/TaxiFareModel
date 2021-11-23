from os import pipe
from TaxiFareModel.data import get_data, clean_data
from sklearn.model_selection import train_test_split
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import numpy as np


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.15)

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())])

        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                          ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
        "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
        'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                     remainder="drop")
        pipe = Pipeline([('preproc', preproc_pipe),
                     ('linear_model', LinearRegression())])
        return pipe

    def run(self):
        """set and train the pipeline"""
        pipeline = self.set_pipeline()
        return pipeline.fit(self.X_train, self.y_train)

    def evaluate(self):
        """evaluates the pipeline on df_test and return the RMSE"""
        trained_model = self.run()
        y_pred = trained_model.predict(self.X_test)
        rmse = np.sqrt(((y_pred - self.y_test)**2).mean())
        #print(rmse)
        #print('eval')
        return rmse


if __name__ == "__main__":
    # get data
    df = get_data()

    # clean data
    df = clean_data(df)

    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    # train
    train = Trainer(X, y)
    train.run()

    # evaluate
    train.evaluate()
