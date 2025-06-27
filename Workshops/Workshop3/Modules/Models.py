from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from xgboost import XGBClassifier

from absl import logging
import tensorflow_decision_forests as tfdf

class ModelML(ABC):
    def __init__(self, VALID_USER_LIST):
        self._prediction_df = pd.DataFrame(
            data=np.zeros((len(VALID_USER_LIST), 18)),
            index=VALID_USER_LIST
        )
        self._models = {}
        self._evaluation_dict = {}

    @abstractmethod
    def train(self, train_x, valid_x, labels):
        pass


class RandomForestTreeModel(ModelML):
    tfdf.keras.get_all_models()
    rf = tfdf.keras.GradientBoostedTreesModel(hyperparameter_template="benchmark_rank1")
    def __init__(self, VALID_USER_LIST):
        super().__init__(VALID_USER_LIST)
        # Configuración de logs para suprimir mensajes no deseados
        logging.set_verbosity(logging.ERROR)
        tfdf.keras.set_training_logs_verbosity(0)

    def train(self, train_x, valid_x, labels):
        for q_no in range(1, 19):
            if q_no <= 3:
                grp = '0-4'
            elif q_no <= 13:
                grp = '5-12'
            else:
                grp = '13-22'

            train_df = train_x.loc[train_x.level_group == grp]
            train_users = train_df.index.values
            valid_df = valid_x.loc[valid_x.level_group == grp]
            valid_users = valid_df.index.values

            train_labels = labels.loc[labels.q == q_no].set_index('session').loc[train_users]
            valid_labels = labels.loc[labels.q == q_no].set_index('session').loc[valid_users]

            train_df = train_df.copy()
            valid_df = valid_df.copy()
            train_df["correct"] = train_labels["correct"]
            valid_df["correct"] = valid_labels["correct"]

            train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
                train_df.drop(columns=['level_group']),
                label="correct"
            )
            valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
                valid_df.drop(columns=['level_group']),
                label="correct"
            )

            gbtm = tfdf.keras.GradientBoostedTreesModel(verbose=0)
            gbtm.compile(metrics=["accuracy"])
            gbtm.fit(x=train_ds)

            self._models[f'{grp}_{q_no}'] = gbtm
            evaluation = gbtm.evaluate(x=valid_ds, return_dict=True)
            self._evaluation_dict[q_no] = evaluation["accuracy"]

            predict = gbtm.predict(x=valid_ds)
            self._prediction_df.loc[valid_users, q_no - 1] = predict.flatten()

        return self._models, self._evaluation_dict


class XGBoostModel(ModelML):
    def train(self, train_x, valid_x, labels):
        for q_no in range(1, 19):
            if q_no <= 3:
                grp = '0-4'
            elif q_no <= 13:
                grp = '5-12'
            else:
                grp = '13-22'

            train_df = train_x[train_x.level_group == grp].copy()
            valid_df = valid_x[valid_x.level_group == grp].copy()
            train_users = train_df.index.values
            valid_users = valid_df.index.values

            train_labels = labels.loc[labels.q == q_no].set_index('session').loc[train_users]["correct"]
            valid_labels = labels.loc[labels.q == q_no].set_index('session').loc[valid_users]["correct"]

            train_df = train_df.drop(columns=["level_group"])
            valid_df = valid_df.drop(columns=["level_group"])

            model = XGBClassifier(
                objective="binary:logistic",
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            model.fit(train_df, train_labels)

            self._models[f"{grp}_{q_no}"] = model
            self._evaluation_dict[q_no] = model.score(valid_df, valid_labels)

            predictions = model.predict(valid_df)
            self._prediction_df.loc[valid_users, q_no - 1] = predictions

        return self._models, self._evaluation_dict