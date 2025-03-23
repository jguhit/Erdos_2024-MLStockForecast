import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier
from xgboost import XGBRegressor
import keras 
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import data_engineering as de 
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error


class StockModels:
    def __init__(self):
        self.lr = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
        self.parameters = {'C': [.001, .01, .1, 1, 10, 100]}
        self.clf = GridSearchCV(self.lr, self.parameters)
        self.dummy = DummyClassifier(strategy='most_frequent')
        self.xgb = XGBRegressor()
        
    def load_data(self, use_roberta=False):
        df_dict = de.separate_by_stock(use_roberta=use_roberta)
        df_dict = de.fillna(df_dict)
        return df_dict
    
    def logistic_regression(self, df_dict, features, val=True):
        """Run Logistic Regression model for validation/testing"""
        cv_trades = [{} for _ in range(4)] if val else {}
        cv_opens = [{} for _ in range(4)] if val else {}
        dumb_trades = [{} for _ in range(4)] if val else {}
        accuracy_scores = []
        rmse_scores = []

        for tick in df_dict:
            train, test = de.train_test_split(df_dict[tick])
            train, test = train[features], test[features]
            # Convert to -1/1 format
            train['y'] = train['y'].apply(lambda x: 1 if x >= 0 else -1)
            test['y'] = test['y'].apply(lambda x: 1 if x >= 0 else -1)

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

                    accuracy = accuracy_score(y_test, predict)
                    rmse = np.sqrt(mean_squared_error(y_test, predict))
                    accuracy_scores.append(accuracy)
                    rmse_scores.append(rmse)

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
                cv_opens[tick] = test["Open"].to_numpy()

                accuracy = accuracy_score(y_test, predict)
                rmse = np.sqrt(mean_squared_error(y_test, predict))
                accuracy_scores.append(accuracy)
                rmse_scores.append(rmse)

        return cv_trades, cv_opens, dumb_trades, accuracy_scores, rmse_scores
    
    def gradient_boosting(self, df_dict, features, val=True):
        """Run Gradient Boosting model for validation/testing"""
        cv_trades_xgb = [{} for _ in range(4)] if val else {}
        cv_opens = [{} for _ in range(4)] if val else {}
        accuracy_scores_xgb = []
        rmse_scores_xgb = []

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

                    model_xgb = self.xgb.fit(X_train_scaled, y_train)
                    predict_xgb = model_xgb.predict(X_test_scaled)

                    predict_xgb = np.sign(predict_xgb)
                    predict_xgb[predict_xgb == 0] = -1

                    cv_trades_xgb[i][tick] = predict_xgb

                    # Calculate accuracy and RMSE for validation set
                    y_test_transformed = y_test.copy()
                    y_test_transformed[y_test_transformed == 0] = -1
                    accuracy_xgb = accuracy_score(y_test_transformed, predict_xgb)
                    rmse_xgb = np.sqrt(mean_squared_error(y_test_transformed, predict_xgb))

                    accuracy_scores_xgb.append(accuracy_xgb)
                    rmse_scores_xgb.append(rmse_xgb)

                    i += 1
            else:
                X_train, y_train = train.drop(columns=['y']), train['y']
                X_test, y_test = test.drop(columns=['y']), test['y']

                scaler = MinMaxScaler()
                scaler.fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model_xgb = self.xgb.fit(X_train_scaled, y_train)
                predict_xgb = model_xgb.predict(X_test_scaled)

                predict_xgb = np.sign(predict_xgb)
                predict_xgb[predict_xgb == 0] = -1

                cv_trades_xgb[tick] = predict_xgb
                cv_opens[tick] = test["Open"].to_numpy()

                # Calculate accuracy and RMSE for test set
                y_test_transformed = y_test.copy()
                y_test_transformed[y_test_transformed == 0] = -1
                accuracy_xgb = accuracy_score(y_test_transformed, predict_xgb)
                rmse_xgb = np.sqrt(mean_squared_error(y_test_transformed, predict_xgb))

                accuracy_scores_xgb.append(accuracy_xgb)
                rmse_scores_xgb.append(rmse_xgb)
        
        return cv_trades_xgb, cv_opens, accuracy_scores_xgb, rmse_scores_xgb
    
    def create_sequences(self, X, y, lookback=5):
        """Convert data into sequences for LSTM processing"""
        X_seq, y_seq = [], []
        for i in range(lookback, len(X)):
            X_seq.append(X[i-lookback:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    def fit_lstm(self, X, y, batch_size, nb_epoch, neurons, lookback=5):
        """Train LSTM model with proper sequence handling"""
        keras.backend.clear_session()
        
        # Create sequences from data
        X_seq, y_seq = self.create_sequences(X, y, lookback)
        
        # Build model
        model = Sequential()
        model.add(LSTM(neurons, input_shape=(lookback, X.shape[1])))
        model.add(Dropout(0.2)) 
        model.add(Dense(1))
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        
        model.fit(
            X_seq, y_seq,
            epochs=nb_epoch,
            batch_size=batch_size,
            verbose=1,
            shuffle=False
        )
      
        return model, lookback
    
    def forecast_lstm(self, model, X, lookback):
        """Generate prediction for a sequence"""
        # Reshape input to a single sequence
        X = X.reshape(1, lookback, X.shape[1])
        yhat = model.predict(X, verbose=0)
        return yhat[0,0]
    
    def run_lstm_model(self, train, test, epochs=10, neurons=5, lookback=1):
        """Run LSTM model pipeline with sequence handling"""
        # Prepare data
        X, y = train.drop(columns=['y']).values, train.y.values
        X_test, y_test = test.drop(columns=['y']).values, test.y.values

        # Scale data
        scaler_X = MinMaxScaler(feature_range=(-1,1))
        scaler_y = MinMaxScaler(feature_range=(-1,1))
        
        X = scaler_X.fit_transform(X)
        y = scaler_y.fit_transform(y.reshape(-1,1)).reshape(-1,)
        X_test = scaler_X.transform(X_test)

        # Train model with sequences
        model, lookback = self.fit_lstm(X, y, batch_size=32, nb_epoch=epochs, neurons=neurons, lookback=lookback)

        # Generate predictions
        predictions = []
        for i in range(lookback, len(X_test)):
            seq = X_test[i-lookback:i]
            yhat = self.forecast_lstm(model, seq, lookback)
            yhat = scaler_y.inverse_transform(np.array([yhat]).reshape(1,1))[0,0]
            predictions.append(yhat)
        
        # Align test data with predictions
        y_test = y_test[lookback:]
        if len(predictions) < len(y_test):
            y_test = y_test[:len(predictions)]
        
        # Calculate metrics using trading signals
        pred_signals = np.sign(predictions)
        pred_signals[pred_signals == 0] = -1
        y_test_signals = np.sign(y_test)
        y_test_signals[y_test_signals == 0] = -1
        
        rmse = np.sqrt(mean_squared_error(y_test_signals, pred_signals))
        accuracy = accuracy_score(y_test_signals, pred_signals)
        
        # Return trading signals
        return np.sign(predictions)/np.abs(np.sign(predictions)), accuracy, rmse
    
    def lstm(self, df_dict, features, val=True):
        """Run LSTM for for validation/testing"""
        cv_trades = [{} for _ in range(4)] if val else {}
        cv_opens = [{} for _ in range(4)] if val else {}
        accuracy_scores = []
        rmse_scores = []

        for tick in df_dict:
            train, test = de.train_test_split(df_dict[tick])
            train, test = train[features], test[features]

            if val:
                i = 0
                for train_idx, test_idx in de.get_cv_splits(train):
                    cv_opens[i][tick] = train.loc[test_idx, "Open"].to_numpy()
                    df_tt = train.loc[train_idx]
                    df_ho = train.loc[test_idx]

                    trades, accuracy, rmse = self.run_lstm_model(df_tt, df_ho)

                    cv_trades[i][tick] = trades
                    accuracy_scores.append(accuracy)
                    rmse_scores.append(rmse)

                    i += 1
            else:
                trades, accuracy, rmse = self.run_lstm_model(train, test)
                cv_trades[tick] = trades
                cv_opens[tick] = test["Open"].to_numpy()

                accuracy_scores.append(accuracy)
                rmse_scores.append(rmse)

        return cv_trades, cv_opens, accuracy_scores, rmse_scores