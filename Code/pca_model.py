
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import ecopy as ep
from sklearn.decomposition import PCA

input_data = pd.read_csv('/home/llevin/Desktop/capstone2/cleaned_data/input_data_vF3.csv')

num_cols = input_data.select_dtypes(include=['int64','float64']).columns
num_cols = [col for col in num_cols if col not in ['result','full_time_home_goals','full_time_away_goals',
                                                   'half_time_home_goals','half_time_away_goals','half_time_results'
                                                   ,'at_Emirates Stadium','at_Boleyn Ground','at_King Power Stadium'
                                                   ,'at_Old Trafford','at_Loftus Road','at_Britannia Stadium'
                                                   ,'at_The Hawthorns','at_Anfield','at_Sports Direct Arena'
                                                   ,'at_Turf Moor','at_Villa Park','at_Stamford Bridge','at_Selhurst Park'
                                                   ,'at_Goodison Park',"at_St Mary's Stadium",'at_Liberty Stadium',
                                                   'at_KC Stadium','at_Stadium of Light','at_White Hart Lane'
                                                   ,'at_Etihad Stadium','at_Carrow Road','at_Molineux Stadium',
                                                   'at_Cardiff City Stadium','at_Craven Cottage','at_Madejski Stadium']]
pca_cols = [col for col in num_cols if col not in ['home_team_api_id','away_team_api_id']]
X = input_data[pca_cols].values

scale = StandardScaler()

norm_data = scale.fit_transform(X)
norm_data.shape

soccer_pca = PCA()
soccer_pca.fit(norm_data)
soccer_pcs = pca.transform(norm_data)
soccer_pcs = pd.DataFrame(soccer_pcs,columns=['PC'+str(i) for i in range(1, soccer_pcs.shape[1]+1)])
soccer_pcs.to_csv('/home/llevin/Desktop/capstone2/cleaned_data/pca_df.csv',index=False)
