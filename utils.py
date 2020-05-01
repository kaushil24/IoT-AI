import pandas as pd
import numpy as np
import heartpy as hp
from datetime import datetime as dt
from biosppy.signals import ecg
from ecgdetectors import Detectors # pip3 install py-ecg-detectors
from hrv import HRV
import neurokit as nk
from biosppy.signals.eda import eda
from pysiology import electrodermalactivity
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Conv2D, Conv1D,MaxPooling2D, UpSampling2D, MaxPooling1D, BatchNormalization, UpSampling1D
from keras.models import Model

class Preprocessing():
    
    def __init__(self, df):
        self.df = df
        self.freq = 500
    
    def ECGfeatures(self,df, sampling_rate = 500):
        drivers = df['Driver'].value_counts()
        df.reset_index(inplace=True)
        # Filtering ECG to remove noise
        for dr in drivers.index:
            after = ecg.ecg(signal = df.loc[df['Driver']== dr, 'ECG'], sampling_rate = sampling_rate, show = False)
            df.loc[df['Driver']== dr, 'ECG'] = after[1]
            
        #Calculating R-R Peaks
        detectors = Detectors(sampling_rate)
        r_peaks = detectors.engzee_detector(df['ECG'])

        # Calculating R-R Interval 
        df['RMSSD'] = [0 for _ in range(len(df))]
        rp = pd.Series(r_peaks)
        hrv_obj = HRV(sampling_rate)
        for dr in drivers.index:
            start = df.loc[df['Driver']==dr].index[0]
            end = df.loc[df['Driver']==dr].index[-1]
            RMSSD = hrv_obj.RMSSD(rp[(rp>start) & (rp<end)], normalise = False)
            df.loc[df['Driver']==dr, 'RMSSD'] = RMSSD
            
        print('Finished extracting RMSSD successfully')

        drivers = df['Driver'].value_counts().keys()
        # Calculating other ECG features
        df['meanNN'] = [0 for _ in range(len(df))]
        df['sdNN'] = [0 for _ in range(len(df))]
        df['cvNN'] = [0 for _ in range(len(df))]
        df['CVSD'] = [0 for _ in range(len(df))]
        df['medianNN'] = [0 for _ in range(len(df))]
        df['madNN'] = [0 for _ in range(len(df))]
        df['mcvNN'] = [0 for _ in range(len(df))]
        df['pNN20'] = [0 for _ in range(len(df))]
        df['pNN50'] = [0 for _ in range(len(df))]
        for dr in drivers:
            start = df.loc[df['Driver']==dr, :].index[0]
            end = df.loc[df['Driver']==dr, :].index[-1]
            crop = rp[(rp>start) & (rp<end)].diff()
            rmssd = df.loc[start, 'RMSSD']
            meanNN = crop.mean()
            df.loc[df['Driver']==dr, 'meanNN'] = meanNN
            std = crop.std()
            df.loc[df['Driver']==dr, 'sdNN'] = std
            cvNN = std/meanNN
            df.loc[df['Driver']==dr, 'cvNN'] = cvNN
            cvsd = rmssd/meanNN
            df.loc[df['Driver']==dr, 'CVSD'] = cvsd
            medianNN = crop.median()
            df.loc[df['Driver']==dr, 'medianNN'] = medianNN
            madNN = crop.mad()
            df.loc[df['Driver']==dr, 'madNN'] = madNN
            mcvNN = madNN / medianNN
            df.loc[df['Driver']==dr, 'mcvNN'] = mcvNN
            pNN20 = hrv_obj.pNN20(crop)
            df.loc[df['Driver']==dr, 'pNN20'] = pNN20
            pNN50 = hrv_obj.pNN50(crop)
            df.loc[df['Driver']==dr, 'pNN50'] = pNN50

        print('Finished extracting meanNN, sdNN, cvNN, CVSD, medianNN, madNN, mcvNN, pNN20, pNN50 successfully')
        self.df = df
        return df

    
    def GSRfeatures(self,df, sampling_rate = 31):
        drivers = df['Driver'].value_counts().index
        df['foot meanGSR'] = [0 for _ in range(len(df))]
        df['hand meanGSR'] = [0 for _ in range(len(df))]
        df['foot meanSCR'] = [0 for _ in range(len(df))]
        df['hand meanSCR'] = [0 for _ in range(len(df))]
        df['foot maxSCR'] = [0 for _ in range(len(df))]
        df['hand maxSCR'] = [0 for _ in range(len(df))]
        df['foot meanSCL'] = [0 for _ in range(len(df))]
        df['hand slopeSCL'] = [0 for _ in range(len(df))]
        for dr in drivers:
            for var in ['foot', 'hand']:
                temp = df.loc[df['Driver']==dr, var+' GSR']
                meanGSR = temp.mean()
                df.loc[temp.index, var+' meanGSR'] = meanGSR
                processed_GSR = nk.eda_process(temp, sampling_rate = sampling_rate)
                meanSCL = processed_GSR['df']['EDA_Tonic'].mean()
                df.loc[temp.index, var+' meanSCL'] = meanSCL
                slopeSCL = max(processed_GSR['df']['EDA_Tonic']) - min(processed_GSR['df']['EDA_Tonic'])
                df.loc[temp.index, var+' slopeSCL'] = slopeSCL
                meanSCR = processed_GSR['df']['EDA_Phasic'].mean()
                df.loc[temp.index, var+' meanSCR'] = meanSCR
                maxSCR = max(processed_GSR['df']['EDA_Phasic'])
                df.loc[temp.index, var+' maxSCR'] = maxSCR
                
        print('Finished extracting foot and hand GSR features: meanGSR, meanSCL, slopeSCL and maxSCR')
        self.df = df
        return df
    
    def train_test(self, df, norm = False, using_CAE = True):
        del_cols = ['Elapsed time', 'Driver', 'marker', 'stress']
        
        if using_CAE:
            del_cols+=['ECG']
            
        X = df.drop(del_cols, axis = 1)
        y = df['stress']
        if norm:
            X=(X-X.min())/(X.max()-X.min()) # Normalising to rescale all the features. 
            
        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_Y = encoder.transform(y)
        trainX, testX, trainy, testy = train_test_split(X, encoded_Y, test_size = 0.2 )
        num_features = len(X.columns)
        return trainX, testX, trainy, testy, num_features
    
    
class Modelling():
    
    def predStressFCNN(self, input_shape):
        '''
        This is the same architecture as proposed by Azar et al.
        '''
        model = Sequential()
        model.add(Dense(60, input_dim=input_shape, activation='relu'))
        model.add(Dense(60, activation = 'relu'))
        model.add(Dense(60, activation = 'relu'))
        model.add(Dense(60, activation = 'relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam' ,metrics=['accuracy'])
        model.summary()
        return model
    
    def CAE(self, time_steps = 2000):
        input_sample = Input(shape=(time_steps, 1))
        encoding = Conv1D(filters=8, kernel_size=3, activation='relu', padding = 'same')(input_sample)
        encoding = MaxPooling1D(pool_size = 2, padding = 'same')(encoding)
        encoding = Conv1D(filters = 32, kernel_size = 5, activation = 'relu', padding = 'same')(encoding)
        encoding = BatchNormalization()(encoding)
        encoding = MaxPooling1D(pool_size = 2, padding = 'same')(encoding)

        encoding = Conv1D(filters = 16, kernel_size = 3, activation = 'relu', padding = 'same')(encoding)
        encoding = BatchNormalization()(encoding)
        encoding = MaxPooling1D(pool_size = 2, padding = 'same')(encoding)

        encoding = Conv1D(filters = 64, kernel_size = 11, activation = 'relu', padding = 'same')(encoding)
        encoding = Conv1D(filters = 128, kernel_size = 13, activation = 'relu', padding = 'same')(encoding)
        encoding = MaxPooling1D(pool_size = 2, padding = 'same')(encoding)
        encoding = Conv1D(filters = 32, kernel_size = 3, activation = 'relu', padding = 'same')(encoding)
        encoding = Conv1D(filters = 1, kernel_size = 7, activation = 'relu', padding = 'same')(encoding)
        encoding = MaxPooling1D(pool_size = 2, padding = 'same')(encoding)

        # Making decoder:
        decoding = Conv1D(filters = 1, kernel_size = 7, padding='same', activation = 'relu')(encoding)
        decoding = Conv1D(32, 3, activation = 'relu', padding = 'same')(decoding)
        decoding = UpSampling1D()(decoding)
        decoding = Conv1D(64, 11, activation = 'relu', padding = 'same')(decoding)
        decoding = Conv1D(128, 13, activation = 'relu', padding = 'same')(decoding)
        decoding = UpSampling1D()(decoding)
        decoding = Conv1D(16, 3, activation = 'relu', padding = 'same')(decoding)
        decoding = Conv1D(32, 5, activation = 'relu', padding = 'same')(decoding)
        decoding = UpSampling1D()(decoding)
        decoding = Conv1D(32, 3, activation = 'relu', padding = 'same')(decoding)
        decoding = UpSampling1D()(decoding)
        decoding = Conv1D(8, 3, activation = 'relu', padding = 'same')(decoding)
        decoding = Flatten()(decoding)
        decoding = Dense(time_steps)(decoding)#, activation = 'sigmoid')(decoding)

        model = Model(input_sample, decoding)
        model.compile(optimizer='adadelta', loss='mean_squared_error')

        # Making a seperate encoder 
        encoder = Model(input_sample, encoding)

        return model, encoder
    
    def enc_columns_CAE(self, df, steps = 2000, epochs = 50):
        cols =  ['ECG', 'EMG', 'foot GSR', 'hand GSR', 'HR', 'RESP']

        for col in cols:
            enc_df = pd.DataFrame(columns= ['Driver',col + ' Encoded'])
            driver = df['Driver'].value_counts().keys()
            df[col+' CAE'] = [-1 for _ in range(len(df))]
            for dr in driver:
                start_index = df.loc[df['Driver']==dr, col].index[0]
                train_CAE = []
                encoding_series = df.loc[df['Driver']==dr, col]
                for val in range(steps, len(encoding_series), steps):
                    temp = np.array(encoding_series[val-steps: val]).reshape((steps, 1))
                    train_CAE.append([temp])

                train_CAE = np.array(train_CAE)
                train_CAE = train_CAE.reshape(-1, steps, 1)
                print(train_CAE.shape)

                mdl, enc = self.CAE(steps)

                mdl.fit(train_CAE, train_CAE.reshape(-1, steps), epochs = epochs, shuffle = False)
                dec_CAE = mdl.predict(train_CAE)
                mdl.save(col + 'CAE Model.h5')
                dec_CAE = dec_CAE.reshape(train_CAE.shape[0] * train_CAE.shape[1])
                dec_CAE = list(dec_CAE)
                end_index = start_index + len(dec_CAE) - 1
            #     filtered_ecg = filtered = ecg.ecg(ecg_CAE, sampling_rate = 500, show = False)
                print(len(dec_CAE), start_index, start_index+len(dec_CAE))
                df.loc[start_index:start_index+len(dec_CAE)-1, col+' CAE'] = dec_CAE #filtered_ecg['filtered']

                enc_CAE= enc.predict(train_CAE)
                enc_CAE = enc_CAE.reshape(63,)

                enc_df.append({'Driver':[dr for _ in range(len(enc_CAE))], col+' Encoded':[val for _ in enc_CAE]})

                print('*'*5, col+':' + dr, 'Done')

            enc_df.to_csv(col+' encoded.csv', index = False)

        return df


if __name__ == "__main__":
    
    df = pd.read_csv('ds01_withStress.csv')
    utils = Preprocessing(df)
    df = utils.ECGfeatures(df)
    df = utils.GSRfeatures(df)
    print(df.columns)