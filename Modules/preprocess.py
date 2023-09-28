import pandas as pd


class preprocess:
     
        def __init__(self,df: pd.DataFrame):
            
            assert "date" in df.columns, "date is not found in axis"
            self.df = df
            self.rows = df.shape[0]
            self.columns = list(df.columns)
                
        def handle_missing_entries(self, freq: str = 'D'):
                    
                    assert 'date' in self.columns,"date is not found in axis"
        
                    list(self.columns).remove('date')
                    self.df['date'] = pd.to_datetime(self.df['date'])
                    date_range = pd.date_range(start=self.df['date'].min(), end=self.df['date'].max(), freq=freq)
                    new_df = pd.DataFrame({'date': date_range})
                    merged_df = pd.concat([self.df.set_index('date'), new_df.set_index('date')], axis=1).reset_index()
                    merged_df[self.columns] = merged_df[self.columns].ffill()
                    return merged_df
            
        def aggregate(self,df: pd.DataFrame, level: str):
                    assert 'date' in self.columns, "date is not found in axis"
                    assert level in ['daily','weekly', 'monthly', 'yearly','None'], "Invalid aggregation level"

                    if level == 'daily':
                        aggregated_df = df.groupby(pd.Grouper(key='date', freq='D')).mean().reset_index()
                    if level == 'weekly':
                        aggregated_df = df.groupby(pd.Grouper(key='date', freq='W')).mean().reset_index()
                    elif level == 'monthly':
                        aggregated_df = df.groupby(pd.Grouper(key='date', freq='M')).mean().reset_index()
                    elif level == 'yearly':
                        aggregated_df = df.groupby(pd.Grouper(key='date', freq='Y')).mean().reset_index()

                    elif level == 'None':
                           aggregated_df = df

                    return aggregated_df
                
        def set_datetime_index(self,df: pd.DataFrame):
                    
                    assert "date" in self.columns, "date is not found in axis"
                    
                    df.index = pd.to_datetime(df['date'])
                    del df['date']
                    return df
        

        def train_test_split(self, df: pd.DataFrame, split: float = 0.7) -> tuple: 
               
                partition = int(len(df)*split)
                df_train = df.iloc[0:partition,:]
                df_test = df.iloc[partition:,:]

                return df_train, df_test
        

# def run_preprocess(ts_data,freq: str):
#           preprocess_util = preprocess(ts_data) 
#           new_df = preprocess_util.handle_missing_entries(freq='M')
#           agg_df = preprocess_util.aggregate(new_df, level='yearly')   
#           final_df = preprocess_util.set_datetime_index(agg_df)     
#           return final_df



         
             