#!/usr/local/bin/python3.7
import streamlit as st
import pandas as pd
import numpy as np
import base64
from matplotlib.pylab import plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, LSTM, GRU
from keras.layers.convolutional import Conv1D
from keras.optimizers import SGD
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
#
@st.cache
def load_data(url='https://raw.githubusercontent.com/KosarevVS/Dockers/master/my_data.csv',userdata=None):
    if userdata is not None:
        try:
            df_init=pd.read_csv(userdata,index_col=0)
            df_init=df_init.dropna()
            df_init.index = pd.to_datetime(df_init.index,errors='coerce')
        except:
            st.write('Что-то не то с форматом данных.')
    else:
        df_init=pd.read_csv(url,index_col=0)
        # df_init=pd.read_csv('testherok.csv',index_col=0)
        # df_init=df_init.dropna()
        # df.mask()#убрать значения больше 3 стд отклонений
        my_dates=pd.date_range(start='2001-01-31',periods=len(df_init),freq='M')
        df_init.index=my_dates
    return df_init
#
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences[:-n_steps])):#последние n_steps не берем, потому что прогнозировать нечего
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
def prep_data(ncol,data_sep,n_steps=6,userdata=None):
    scl = StandardScaler()
    if userdata is not None:
        data=load_data(userdata=userdata).values[:,0]
    else:
        data=load_data(userdata=userdata).values[:,ncol]
    df_init_scal = scl.fit_transform(data.reshape(data.shape[0], 1))#сохраняем мат и дисп для скал тестовой выборки
    a=df_init_scal
    b=np.roll(df_init_scal,-1)
    X,y=split_sequences(np.hstack([a,b]), n_steps)
    sep=round(data_sep*len(data))
    X_train=X[:sep-n_steps]
    X_test=X[sep-n_steps:]
    y_train=y[:sep-n_steps]
    y_test=y[sep-n_steps:]
    ytest_or=data[sep:]
    return X_train,X_test,y_train,y_test,scl,ytest_or

#
class select_model():
    """
    Выбор модели и получение прогноза, отрисовка процесса обучения и результатов прогноза
    """
    def __init__(self,x_train,y_train,x_test,y_test,n_steps=6,n_epoh=200):
        self.n_steps  = n_steps
        self.n_epoh  = n_epoh
        self.x_train  = x_train
        self.y_train  = y_train
        self.x_test   = x_test
        self.y_test   = y_test
        # assert(len(x_train)==len(y_train)and len(x_test)==len(y_test))
    def plot_his(self,hist):
        fig1,ax1 = plt.subplots(figsize=(18, 5))
        ax1.plot(hist.history['mean_squared_error'],label='Обучение')
        ax1.plot(hist.history['val_mean_squared_error'],label='Валидация')
        ax1.legend(loc='best',fontsize=16)
        ax1.grid()

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
              {'out_1':self.y_test}),epochs=self.n_epoh, batch_size=len(self.x_test), verbose=0)
        plottr=self.plot_his(hist=history)
        st.subheader("Процесс обучения")
        st.pyplot(fig=plottr, clear_figure=True, use_container_width=True)
        return model

    def simple_gru(self):
        ipp_1 = Input(shape=(self.n_steps, self.x_train.shape[2]),name='fact_ipp_1')
        gru1=GRU(6, activation='relu', input_shape=(self.n_steps, self.x_train.shape[2]))(ipp_1)
        ipp_1_pred=Dense(1,activation='linear', name='out_1')(gru1)
        model = Model([ipp_1],[ipp_1_pred])
        model.compile(optimizer='adam',
                      loss={'out_1': 'mse'},
                    metrics=['mse', 'mae', 'mape'])
        history=model.fit({'fact_ipp_1': self.x_train},{'out_1':self.y_train},validation_data=({'fact_ipp_1': self.x_test},
              {'out_1':self.y_test}),epochs=self.n_epoh, batch_size=len(self.x_test), verbose=0)
        ###################
        plottr=self.plot_his(hist=history)
        st.subheader("Процесс обучения")
        st.pyplot(fig=plottr, clear_figure=True, use_container_width=True)
        # my_predicts=model.predict(self.x_test).flatten()
        return model

    def simple_cnn(self):
        ipp_1 = Input(shape=(self.n_steps, self.x_train.shape[2]),name='fact_ipp_1')
        cnn1 = Conv1D(filters=1, kernel_size=3, activation='relu')(ipp_1)
        merge = Conv1D(filters=15, kernel_size=1, activation='relu')(cnn1)
        merge = Flatten()(merge)
        ipp_1_pred=Dense(1,activation='linear', name='out_1')(merge)
        model = Model([ipp_1],[ipp_1_pred])
        model.compile(optimizer='adam',
                      loss={'out_1': 'mse'},
                    metrics=['mse', 'mae', 'mape'])
        history=model.fit({'fact_ipp_1': self.x_train},{'out_1':self.y_train},validation_data=({'fact_ipp_1': self.x_test},
              {'out_1':self.y_test}),epochs=self.n_epoh, batch_size=len(self.x_test), verbose=0)
        ###################
        plottr=self.plot_his(hist=history)
        st.subheader("Процесс обучения")
        st.pyplot(fig=plottr, clear_figure=True, use_container_width=True)
        # my_predicts=model.predict(self.x_test).flatten()
        return model

#
def plot_forec_val(ytest,predict,my_dates,scl,y_hat):
    date_app=pd.date_range(start=my_dates[-1],periods=len(y_hat),freq='M')
    fig2,ax2 = plt.subplots(figsize=(18, 5))
    ax2.plot(my_dates,ytest, color='black', label = 'Факт')
    ax2.plot(my_dates,scl.inverse_transform(predict),'-.', color='blue', label = 'Прогноз')
    ax2.plot(date_app,scl.inverse_transform(y_hat),'-.', color='blue')
    ax2.legend(loc='best',fontsize=16)
    ax2.grid()

def forec_per(model,x_test,forec_per):
    x=x_test[-1]
    y_hat=np.array([])
    for i in range(forec_per):
        y_un=model.predict(x.reshape(1,6,1))[0]
        y_hat=np.hstack([y_hat,y_un])
        x=np.concatenate((x,y_un.reshape(-1,1)))[-x_test[-1].shape[0]:]
    return y_hat

def print_rez(model,x_test,y_test,per_forecast,scl,ytest_or,userdata,data_sep):
    my_predicts=model.predict(x_test).flatten()
    y_hat=forec_per(model,x_test,per_forecast+1)
    sep_date=load_data(userdata=userdata).index[round(data_sep*len(load_data(userdata=userdata)))]
    my_dates = pd.date_range(start=sep_date,periods=len(y_test),freq='M')
    plotfr=plot_forec_val(ytest_or,my_predicts,my_dates,scl,y_hat)
    st.subheader("Прогноз на тесте")
    st.pyplot(fig=plotfr, clear_figure=True, use_container_width=True)
    my_mse=round(metrics.mean_squared_error(y_test, my_predicts),per_forecast+1)
    st.write('Ошибка прогноза на тестовой выборке:', str(my_mse))
    all_forec=np.hstack([y_test[:-1],y_hat])
    a=pd.Series(scl.inverse_transform(all_forec),index=pd.date_range(start=sep_date,periods=len(all_forec),freq='M'))
    b=pd.Series(ytest_or,index=pd.date_range(start=sep_date,periods=len(ytest_or),freq='M'))
    df=pd.concat([b,a],axis=1)
    df.columns=['Факт','Прогноз']
    st.dataframe(df.T)
    csv_exp = df.to_csv()
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Скачать прогноз</a> (добавьте к загруженному файлу расширение **.csv**)'
    st.markdown(href, unsafe_allow_html=True)
    st.success('Done!')
#
def best_arima(train, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for d in d_values:
        result = adfuller(train)
        print(result)
        if round(result[1],3) < 0.05:
            d = 0
        else:
            d = d
        for p in p_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    model = ARIMA(train, order = order)
                    model_fit = model.fit(disp=0)
                    bic = model_fit.bic
                    if bic < best_score:
                        best_score, best_cfg = bic, order
                    # print('ARIMA%s BIC=%.3f' % (order, bic))
                except:
                    continue
    text='Наилучшая ARIMA%s BIC=%.3f' % (best_cfg, best_score)
    return text,best_cfg

def main():
    st.sidebar.header('Параметры построения прогноза')
    userdata=None
    tickdic=dict(zip(load_data(userdata=None).columns,range(0,3)))
    name_fact = st.sidebar.selectbox('Выбор показателя', load_data(userdata=None).columns)
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
    st.subheader('Исходные данные')

    agree = st.checkbox('Згрузить собственные данные')
    if agree:
        userdata = st.file_uploader('Поместитн csv файл сюда. Файл должен содержать два столбца: первый - даты, желательно в формате "YYYY-MM-DD", второй - числовой ряд.', type='csv', encoding='auto')

    if userdata is not None:
        st.line_chart(load_data(userdata=userdata),height=200)
        st.write('Источник данных: Пользователь')
    else:
        st.line_chart(load_data(userdata=userdata)[name_fact],height=200)
        st.write('Источник данных: Росстат')
    st.subheader('Параметры модели')
    per_forecast = st.slider('Выберите прогнозный период (кол-во месяцев):', 1, 12, 12)
    if per_forecast==1:
        st.write("Прогноз показателя будет построет на ", per_forecast, 'месяц вперед')
    elif per_forecast==2 or per_forecast==3 or per_forecast==4:
        st.write("Прогноз показателя будет построет на ", per_forecast, 'месяца вперед')
    else:
        st.write("Прогноз показателя будет построет на ", per_forecast, 'месяцев вперед')
    # d = st.date_input("Выбирите дату разделения данных",datetime.date(2019, 7, 6))
    # st.write('Выбранная дата:', d)
    data_sep=st.slider('Доля тренировочной выборки от исходного набора данных,%:', 1, 100, 85)/100
    ###########################################################################################################

    ##########################################################################################################
    if lstm:
        nneur = st.slider('Количество нейронов на внутреннем слое:', 1, 15, 6)
        wind = st.slider('Размерность паттерна (величина временного окна):', 3, 24, 6)
        # st.write("Прогноз на t+1 период определяют ", wind, ' предыдущих значений прогнозируемого показателя.')
        nepoh = st.slider('Количество эпох обучения:', 50, 200, 200,step=25)
        # a = st.checkbox("Показать описание параметров")
        # if a:
        #     st.write('Величина временного окна - количество наблюдений прогнозируемого показателя, используемых в качестве одного паттерна нейронной сети.')
        # else:
        #     pass
        agree = st.button('Запустить расчет')
        if agree:
            with st.spinner('Идет обучение нейронной сети...'):
                x_train,x_test,y_train,y_test,scl,ytest_or=prep_data(ncol=tickdic[name_fact],data_sep=data_sep,userdata=userdata)
                model=select_model(x_train,y_train,x_test,y_test,6,nepoh).simple_lstm()
                print_rez(model,x_test,y_test,per_forecast,scl,ytest_or,userdata,data_sep)
    if gru:
        nneur = st.slider('Количество нейронов на внутреннем слое:', 1, 15, 6)
        wind = st.slider('Величина временного окна:', 3, 24, 6)
        nepoh = st.slider('Количество эпох обучения:', 50, 200, 150,step=25)
        agree = st.button('Запустить расчет')
        if agree:
            with st.spinner('Идет обучение нейронной сети...'):
                x_train,x_test,y_train,y_test,scl,ytest_or=prep_data(ncol=tickdic[name_fact],data_sep=data_sep,userdata=userdata)
                model=select_model(x_train,y_train,x_test,y_test,6,nepoh).simple_gru()
                print_rez(model,x_test,y_test,per_forecast,scl,ytest_or,userdata,data_sep)
    if cnn:
        nneur = st.slider('Количество сверточных фильтров:', 1, 15, 5)
        wind = st.slider('Величина временного окна:', 3, 24, 6)
        nepoh = st.slider('Количество эпох обучения:', 50, 200, 150,step=25)
        agree = st.button('Запустить расчет')
        if agree:
            with st.spinner('Идет обучение сверточной нейронной сети...'):
                x_train,x_test,y_train,y_test,scl,ytest_or=prep_data(ncol=tickdic[name_fact],data_sep=data_sep,userdata=userdata)
                model=select_model(x_train,y_train,x_test,y_test,6,nepoh).simple_cnn()
                print_rez(model,x_test,y_test,per_forecast,scl,ytest_or,userdata,data_sep)

    if arima:
        agree = st.button('Запустить расчет')
        if agree:
            with st.spinner('Выбор наилучшей ARIMA модели...'):
                # загрузка данных
                if userdata is not None:
                    data=load_data(userdata=userdata)
                else:
                    data=load_data(userdata=userdata)[name_fact]
                # скалирование

                # разделение на трейн и тесте
                train,test=data.values[:round(data_sep*len(data))].flatten(),data.values[round(data_sep*len(data)):].flatten()
                # выбор наилучшей модели на трейне
                text,best_model=best_arima(train, range(0, 4), range(0, 2), range(0, 4))
                # показать процесс выбора наилучшей модели aic pdq, тест Дики фуллера
                st.write(text)

                # итерационный прогноз на тесте и показ результатов
                p ,d,q= best_model[0],best_model[1],best_model[2]
                y_hat=np.array([])
                history=train
                for t in range(len(test)):
                    model = ARIMA(history, order=(1,0,0))#должно стоять q
                    model_fit = model.fit(disp=0)
                    y_hat=np.append(y_hat,model_fit.forecast()[0][0])
                    history=np.append(history,test[t])
                my_df=pd.DataFrame({'y_hat':y_hat,
                               'test':test})
                # построение прогноза на количество прогнозируемых периодов от последней даты
                y_hat2=np.array([])
                history=data.values
                for t in range(per_forecast):
                    model = ARIMA(history, order=(1,0,0))
                    model_fit = model.fit(disp=0)
                    onestep=model_fit.forecast()[0][0]
                    y_hat2=np.append(y_hat2,onestep)
                    history=np.append(history,onestep)
                # print(y_hat2)

                y_hat_con=np.append(y_hat,y_hat2)

                sep_date=load_data(userdata=userdata).index[round(data_sep*len(load_data(userdata=userdata)))]
                my_dates=pd.date_range(start=sep_date,periods=len(y_hat),freq='M')
                date_app=pd.date_range(start=sep_date,periods=len(y_hat_con),freq='M')

                fig,ax = plt.subplots(figsize=(18, 5))
                ax.plot(my_dates,test, color='black', label = 'Факт')
                ax.plot(date_app,y_hat_con,'-.', color='blue', label = 'Прогноз')
                ax.legend(loc='best',fontsize=16)
                ax.grid()
                # вывод графика и df
                st.subheader("Прогноз на тесте")
                st.pyplot(fig=fig, clear_figure=True, use_container_width=True)
                my_mse=round(metrics.mean_squared_error(test, y_hat),per_forecast+1)
                st.write('Ошибка прогноза на тестовой выборке:', str(my_mse))


                a=pd.Series(y_hat_con,index=pd.date_range(start=sep_date,periods=len(y_hat_con),freq='M'))
                b=pd.Series(test,index=pd.date_range(start=sep_date,periods=len(test),freq='M'))
                df=pd.concat([b,a],axis=1)
                df.columns=['Факт','Прогноз']
                st.dataframe(df.T)
                csv_exp = df.to_csv()
                b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
                href = f'<a href="data:file/csv;base64,{b64}">Скачать прогноз</a> (добавьте к загруженному файлу расширение **.csv**)'
                st.markdown(href, unsafe_allow_html=True)
                st.success('Done!')




        ##########################################################################


    #
if __name__ == '__main__':
    main()
