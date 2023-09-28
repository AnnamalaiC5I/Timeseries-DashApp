import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import StandardScaler

import joblib
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import StandardScaler

import joblib
import matplotlib.pyplot as plt

class LSTMFUNC:
            def __init__(self, df: pd.DataFrame):
            
                    assert 'date' in df.columns or 'DATE' in df.columns, "date column must be in a dataframe"
                    self.df = df
                    self.rows = df.shape[0]
                    self.cols = df.shape[1]
                    
            def create_model_lstm(self, n_steps: int, num_layers: int, num_units: list, dropout_rate: float=0.0) -> Sequential:

                        model = Sequential()
                        for i in range(num_layers):
                            if i == 0:
                                model.add(LSTM(units=num_units[i], return_sequences=True, input_shape=(n_steps, self.cols - 1)))
                            else:
                                model.add(LSTM(units=num_units[i], return_sequences=i < num_layers - 1))
                            if dropout_rate !=0.0:
                                 model.add(Dropout(dropout_rate))
                        model.add(Dense(1))
                        model.compile(optimizer='adam', loss='mean_squared_error')

                        self.model = model
                        return model 
                    
            def train_model(self,x: np.ndarray,y: np.ndarray,epochs: int=100,batch_size: int=32) -> Sequential:
            
                    self.model.fit(x,y,epochs=epochs,batch_size=batch_size)
                    return self.model  
                
            # def test_plot(self,df: pd.DataFrame,var1: str,var2:str) -> None:
                    
            #         plt.plot(df[var1],label='actual',color='red')
            #         plt.plot(df[var2],label='predicted',color='blue')
            #         plt.xlabel('Date')
            #         plt.ylabel('Value')
            #         plt.title('Actual vs Predicted')
            #         plt.xticks(rotation=90)
            #         plt.legend()
            #         plt.show()
                    


class Multivariate(LSTMFUNC):
            
            def standardization(self,n_steps: int,train: pd.DataFrame,test: pd.DataFrame,target: str)-> tuple:
                    
                    self.target = target
                    train = train.set_index('date')
                    test = test.set_index('date')
                    print(train.head())
                    sc = StandardScaler()
                    df2_train_scaled = sc.fit_transform(train)
                    
                    joblib.dump(sc, 'scaler_features.joblib')
                    
                    sc2 = StandardScaler()
                    df2_train_scaled_y = sc2.fit_transform(train[[target]])
                    
                    joblib.dump(sc2, 'scaler_target.joblib')
                    
                    df1_train_last14 = train.iloc[-n_steps:]
                    
                    df1_test_full = test.copy()
                    
                    full_df = pd.concat((df1_train_last14,df1_test_full),axis=0)
                    
                    full_df = full_df.reset_index(drop=True)
                    
                    full_df_t = sc.transform(full_df)
                    
                    columns_feature_list = list(train.columns)

                    return df2_train_scaled, df2_train_scaled_y, full_df_t, columns_feature_list
                                                    

            def sampling(self,n_steps: int,df_x: pd.DataFrame,df_y: pd.DataFrame,x_test: np.ndarray) -> tuple: #TODO:
                    X_train = []
                    Y_train = []
                    rows = df_x.shape[0]
                    for i in range(n_steps,rows):
                            X_train.append(df_x[i-n_steps:i])
                            Y_train.append(df_x[i][0])
                            
                    X_train_pred = []
                    test_rows = x_test.shape[0]
                    for i in range(n_steps,test_rows):
                         X_train_pred.append(x_test[i-n_steps:i])
                    X_train_pred = np.array(X_train_pred)
                            
                    return np.array(X_train), np.array(Y_train), np.array(X_train_pred)   
                                    
            def predictions(self,x: np.ndarray) ->np.ndarray:
                
                y_test = self.model.predict(x)
                
                sc2 = joblib.load('scaler_target.joblib')
                
                y_final_pred = sc2.inverse_transform(y_test)
                
                return y_final_pred
            
            def inverse_transform(self,test_df: np.ndarray) -> np.ndarray:
                
                sc = joblib.load('scaler_features.joblib')
                
                test_df = sc.inverse_transform(test_df)
                return test_df
            
            def output_df(self,test_df: pd.DataFrame,column_names: list,prediction: np.ndarray) -> pd.DataFrame:
                
                    #final_df = pd.DataFrame(xtest,columns=column_names)
                    df_pred = test_df.reset_index(drop=True)
                    df_pred[f'{self.target}_predicted'] = prediction
                    
                    return df_pred
                
            
                    
                    
class Univariate(LSTMFUNC):
    
    def sampling(self, sequence: list, n_steps: int) -> tuple:
       
        
        assert isinstance(sequence, list), "must be a list"
        assert isinstance(n_steps, int),   "must be an integer"
        
        X, Y = list(), list()
        
        for i in range(len(sequence)):
            sam = i + n_steps
            if sam > len(sequence)-1:
                break
            x, y = sequence[i:sam], sequence[sam]
            X.append(x)
            Y.append(y)
        
        self.X = X
        self.Y = Y
        self.X_arr = np.array(X)
        self.Y_arr = np.array(Y)
        return np.array(X), np.array(Y), X, Y
    
    
    def create_processed_df(self,n_steps: int) -> pd.DataFrame:
           
            
            assert isinstance(n_steps, int), "must be an integer"
            
            num_columns = len(self.X[0])
            column_names = ['Column' + str(i) for i in range(num_columns)]
            
            try:
                date = self.df['date']
                
            except:
                 date = self.df['DATE']

            # Create DataFrame
            df = pd.DataFrame()
            df = pd.DataFrame(self.X, columns=column_names)
            df['date'] = date[:-n_steps]
            df['target'] = self.Y

            df.set_index('date',inplace=True)
            
            self.preprocessed_df = df
            
            return df

            
    def output_df(self,n_days: int):
           
            assert isinstance(n_days, int), "must be an integer"
            
            df = self.preprocessed_df.copy()
            df2 = df.reset_index('date')
            df2 = df2[['date','target']]
            df2 = df2.iloc[-n_days:,:]
            df2['predicted']= self.result
            
            df2 = df2.set_index('date')
            
            return df2
        
    
    def predictions(self,dfs: pd.DataFrame, model2: Sequential, n_input: int, n_days: int) -> list:
      
        
        assert isinstance(dfs, pd.DataFrame), "must be a Dataframe"
        assert isinstance(model2, Sequential), "must be a Sequential model"
        assert isinstance(n_input, int), "must be an integer"
        assert isinstance(n_days, int), "must be an integer"
        
        test_predictions = []
        
        # Retrieve the last row
        last_row = dfs.iloc[-n_days].tolist()[:-1]
        
        print('last row: {}'.format(last_row))
        
        first_eval_batch = last_row
        current_batch = np.array(first_eval_batch).reshape((1, n_input, 1))
        
        for i in range(0, n_days):
            # get the prediction value for the first batch
            current_pred = model2.predict(current_batch)[0]
            print(current_pred)
            
            # append the prediction into the array
            test_predictions.append(current_pred)
            
            # use the prediction to update the batch and remove the first value
            current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
        
        result = [x[0] for x in test_predictions]
        
        self.result = result
        
        return result
                   