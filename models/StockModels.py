import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier
import data_engineering as de 

class StockModels:
    def __init__(self):
        self.lr = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
        self.parameters = {'C': [.001, .01, .1, 1, 10, 100]}
        self.clf = GridSearchCV(self.lr, self.parameters)
        self.dummy = DummyClassifier(strategy='most_frequent')

    def load_data(self, use_roberta=False):
        df_dict = de.separate_by_stock(use_roberta=use_roberta)
        df_dict = de.fillna(df_dict)
        return df_dict
    
    def logistic_regression(self, df_dict, features, val=True):
        cv_trades = [{} for _ in range(4)] if val else {}
        cv_opens = [{} for _ in range(4)] if val else {}
        dumb_trades = [{} for _ in range(4)] if val else {}
        y_test_val = [{} for _ in range(4)] if val else {}

        for tick in df_dict:
            train, test = de.train_test_split(df_dict[tick])
            train, test = train[features], test[features]
            train['y'] = train['y'].apply(lambda x: 1 if x >= 0 else 0)
            test['y'] = test['y'].apply(lambda x: 1 if x >= 0 else 0)

            if val:
                i = 0
                for train_idx, test_idx in de.get_cv_splits(train):
                    cv_opens[i][tick] = train.loc[test_idx, "Open"].to_numpy()
                    df_tt = train.loc[train_idx]
                    df_ho = train.loc[test_idx]
                    X_train, y_train = df_tt.drop(columns=['y']), df_tt['y']
                    X_test, y_test = df_ho.drop(columns=['y']), df_ho['y']

                    scaler = MinMaxScaler()
                    scaler.fit(X_train)
                    X_train_scaled = scaler.transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    model = self.clf.fit(X_train_scaled, y_train)
                    dumb = self.dummy.fit(X_train_scaled, y_train)

                    predict = model.predict(X_test_scaled)
                    predict[predict == 0] = -1

                    pred_dumb = dumb.predict(X_test_scaled)
                    pred_dumb[pred_dumb == 0] = -1

                    cv_trades[i][tick] = predict
                    dumb_trades[i][tick] = pred_dumb
                    y_test_val[i][tick] = y_test

                    i += 1
            else:
                X_train, y_train = train.drop(columns=['y']), train['y']
                X_test, y_test = test.drop(columns=['y']), test['y']

                scaler = MinMaxScaler()
                scaler.fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = self.clf.fit(X_train_scaled, y_train)
                dumb = self.dummy.fit(X_train_scaled, y_train)

                predict = model.predict(X_test_scaled)
                predict[predict == 0] = -1

                pred_dumb = dumb.predict(X_test_scaled)
                pred_dumb[pred_dumb == 0] = -1

                cv_trades[tick] = predict
                dumb_trades[tick] = pred_dumb
                y_test_val[tick] = y_test.to_numpy()
                cv_opens[tick] = test["Open"].to_numpy()

        return cv_trades, cv_opens, dumb_trades, y_test_val






