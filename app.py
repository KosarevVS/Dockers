#!/usr/local/bin/python3.7
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib.pylab import plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, LSTM, GRU
from keras.layers.convolutional import Conv1D
from keras.optimizers import SGD
#
@st.cache
def load_data(url='https://raw.githubusercontent.com/KosarevVS/stacks-for-TS/master/heroku_app/my_data.csv'):
    df_init=pd.read_csv(url)
    df_init=df_init[['CA','DF','DG']].dropna()
    my_dates=pd.date_range(start='2001-01-31',periods=len(df_init),freq='M')
    df_init.index=my_dates
    return df_init
#
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
    # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
#
def prep_data(ncol):
    scl = StandardScaler()
    df_init_scal = scl.fit_transform(load_data().values)#сохраняем мат и дисп для скал тестовой выборки
    a=df_init_scal[:,ncol].reshape(230,1)
    b=np.roll(df_init_scal[:,ncol],-1).reshape(230,1)
    return np.hstack([a,b])
#
def prep_data_2(stacked,n_steps=6,data_sep=200):
        X,y=split_sequences(stacked, n_steps)
        X_train=X[:data_sep-n_steps+2]
        X_test=X[data_sep-n_steps+1:]
        y_train=y[:data_sep-n_steps+2]
        y_test=y[data_sep-n_steps+1:]
        return X_train,X_test,y_train,y_test
#
class select_model():
    """
    Выбор модели и получение прогноза, отрисовка процесса обучения и результатов прогноза
    """
    def __init__(self,x_train,y_train,x_test,y_test,n_steps=6):
        self.n_steps  = n_steps
        self.x_train  = x_train
        self.y_train  = y_train
        self.x_test   = x_test
        self.y_test   = y_test
        self.my_dates = pd.date_range(start='2017-09-30',periods=len(self.y_test),freq='M')
        # assert(len(x_train)==len(y_train)and len(x_test)==len(y_test))

    def plot_his(self,hist):
        fig1,ax1 = plt.subplots(figsize=(18, 5))
        ax1.plot(hist.history['mean_squared_error'],label='Обучение')
        ax1.plot(hist.history['val_mean_squared_error'],label='Валидация')
        ax1.legend(loc='best',fontsize=16)
        ax1.grid()

    def plot_forec_val(self,predict):
        fig2,ax2 = plt.subplots(figsize=(18, 5))
        # scl = StandardScaler()
        # df_init_scal = scl.fit_transform(load_data().values)#сохраняем мат
        # pred_inv=scl.inverse_transform(self.y_test)
        ax2.plot(self.my_dates,self.y_test, color='black', label = 'Факт')
        ax2.plot(self.my_dates,predict,'-.', color='blue', label = 'Прогноз')
        ax2.legend(loc='best',fontsize=16)
        ax2.grid()

    def simple_lstm(self):#,ytrain,xtest,ytest):
        ipp_1 = Input(shape=(self.n_steps, self.x_train.shape[2]),name='fact_ipp_1')
        lstm1=LSTM(6, activation='relu', input_shape=(self.n_steps, self.x_train.shape[2]))(ipp_1)
        ipp_1_pred=Dense(1,activation='linear', name='out_1')(lstm1)
        model = Model([ipp_1],[ipp_1_pred])
        optim=SGD(momentum=0.01, nesterov=True)
        model.compile(optimizer=optim,
                      loss={'out_1': 'mse'},
                    metrics=['mse', 'mae', 'mape'])
        history=model.fit({'fact_ipp_1': self.x_train},{'out_1':self.y_train},validation_data=({'fact_ipp_1': self.x_test},
              {'out_1':self.y_test}),epochs=200, batch_size=len(self.x_test), verbose=0)
        ###################

        plottr=self.plot_his(hist=history)
        st.subheader("Процесс обучения")
        st.pyplot(fig=plottr, clear_figure=True, use_container_width=True)

        my_predicts=model.predict(self.x_test).flatten()
        plotfr=self.plot_forec_val(predict=my_predicts)
        st.subheader("Прогноз на тесте")
        st.pyplot(fig=plotfr, clear_figure=True, use_container_width=True)

        my_mse=round(metrics.mean_squared_error(self.y_test, my_predicts),3)
        st.write('Ошибка прогноза на тестовой выборке:', str(my_mse))

        df = pd.DataFrame({'Фактические данные':self.y_test,
            'Прогнозные данные':my_predicts})
        df.index=self.my_dates
        st.dataframe(df.T)



    def simple_gru(self):
        print('x')

    def simple_cnn(self):
        print('x')


def main():

    st.sidebar.header('Параметры построения прогноза')
    #
    tickdic=dict(zip(load_data().columns,range(0,3)))
    company_name = st.sidebar.selectbox('Выбор показателя', load_data().columns)
    plot_types = st.sidebar.radio("Выбор прогнозной модели",
        ['LSTM','GRU','CNN','ARIMA'])
    # call the above function
    lstm = (plot_types=='LSTM')
    gru = (plot_types=='GRU')
    cnn = (plot_types=='CNN')
    arima = (plot_types=='ARIMA')
    st.sidebar.title("Описание")
    st.sidebar.info(
    "Это тестовый сайт для прогнозирования основных макроэкономических показателей при помощи\
     моделей искуственных нейронных сетей и классической эконометрики. \n  Код доступен на [github](https://github.com/KosarevVS/Dockers),\
      почта kosarevvladimirserg@gmail.com")

    x_train,x_test,y_train,y_test=prep_data_2(prep_data(tickdic[company_name]))
    my_select_model=select_model(x_train,y_train,x_test,y_test,6)
    if lstm:
        with st.spinner('Идет обучение нейронной сети...'):
            my_select_model.simple_lstm()
    if gru:
        with st.spinner('Идет обучение нейронной сети...'):
            my_select_model.simple_gru()
    if cnn:
        with st.spinner('Идет обучение нейронной сети...'):
            my_select_model.simple_cnn()
    if arima:
        with st.spinner('Построение ARIMA прогноза...'):
            pass
    st.success('Done!')





if __name__ == '__main__':
    main()
