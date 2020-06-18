# define library list
import re
import nltk
import json
import spacy
import string
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import plotly
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt 
import textblob
from textblob import TextBlob
import wordcloud
from wordcloud import WordCloud
from datetime import date
from nltk.corpus import stopwords
from flask import Flask, render_template,request,jsonify
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from IPython.display import display, HTML
from spacy.lang.en.stop_words import STOP_WORDS
from gensim.parsing.preprocessing import remove_stopwords

# Read the COVID-19 data
df = pd.read_csv('covid.csv')
dfT = pd.read_csv('twitter_cleaned.csv')

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(["Location", "Date"], ascending = (True, True))
df['NewConfirmed'] = df.groupby(["Location"])['Confirmed'].diff().fillna(0)
df['NewConfirmed'] = df['NewConfirmed'].astype(int)
df['NewDeaths'] = df.groupby(["Location"])['Deaths'].diff().fillna(0)
df['NewDeaths'] = df['NewDeaths'].astype(int)
df['NewRecovered'] = df.groupby(["Location"])['Recovered'].diff().fillna(0)
df['NewRecovered'] = df['NewRecovered'].astype(int)
recent_date = df['Date'].max()
earliest_date = df['Date'].min()
# Get the latest data
df1 = df[(df['Date'] == recent_date)]
# Remove the countries with zero cases
table1 = df1.drop(df1[df1.Confirmed == 0].index)
# Sort the list based on confirmed case in descending order
table1 = table1.drop(['Text','Date'], axis=1,errors='ignore').sort_values('Confirmed', ascending = False)    
# Remove the worldwide record 
df1 = df1[(df1['Location'] != 'Worldwide')]
countries = countries = table1[(table1['Location'] != 'Worldwide')].sort_values('Location', ascending = True).Location.unique()


app = Flask(__name__)

@app.route('/')
def index():
    feature = 'Worldwide'
    graphs = []
    figs = []
    graphs.append(create_plot(feature,'WorldMap'))
    graphs.append(create_plot(feature,'TopConfirmed'))
    graphs.append(create_plot(feature,'TopNew'))
    graphs.append(create_plot(feature,'TopDeath'))
    graphs.append(create_plot(feature,'TopConfirmedSeries'))
    graphs.append(create_plot(feature,'OverallCase'))
    graphs.append(create_plot(feature,'ActiveCase'))
    figs = create_prediction(feature) 
    urls=plot_twitter_analysis()
    table=table1.to_html(classes='data', header="true", index = False)
    return render_template('index.html', table=table, plots=graphs, pred=figs, countries=countries, urls=urls)

def create_plot(feature,gType):

    df2 = df[df['Location'] == feature]    

    if (gType == 'WorldMap'):
        #display the heat map of latest data
        df1['Text'] = df1['Location'] + '<br>' + \
            'Total Cases :' + df1['Confirmed'].astype(str) + '<br>' + \
            'Recovered :' + df1['Recovered'].astype(str) + '<br>' + \
            'Deaths :' + df1['Deaths'].astype(str)

        fig = go.Figure(data=go.Choropleth(
            locations=df1['Location'],
            locationmode = "country names",
            z=df1['Confirmed'].astype(float),
            colorscale='Reds',
            autocolorscale=False,
            text=df1['Text'], # hover text
            marker_line_color='white' # line markers between states
        ))

        fig.update_layout(margin=dict(
                                l=0,
                                r=0,
                                b=0,
                                t=50,
                                pad=4
                            ),
            title_text='Worldwide COVID-19 Cases (Hover for breakdown)'
        )  
        
    elif (gType == 'TopConfirmed'):
        fig = px.bar(df1.sort_values(by=['Confirmed'], ascending=False)[:10],
             x='Confirmed', y='Location',
             title=f'Top 10 countries with Confirmed cases on {recent_date.strftime("%m-%d-%Y")}', text='Confirmed', orientation='h')
    
    elif (gType == 'TopNew'):
        fig = px.bar(df1.sort_values(by=['NewConfirmed'], ascending=False)[:10],
             x='NewConfirmed', y='Location',
             title=f'Top 10 countries with New cases on {recent_date.strftime("%m-%d-%Y")}', text='NewConfirmed', orientation='h')
             
        fig.update_traces(marker_color='lightsalmon')
             
    elif (gType == 'TopDeath'):        
        fig = px.bar(df1.sort_values(by=['Deaths'], ascending=False)[:10],
             x='Deaths', y='Location',
             title=f'Top 10 countries with Death cases on {recent_date.strftime("%m-%d-%Y")}', text='Deaths', orientation='h')
             
        fig.update_traces(marker_color='indianred')
        
    elif (gType == 'TopConfirmedSeries'):    
        country = df1.sort_values(by=['Confirmed'], ascending=False)[:10]['Location']
        temp = pd.merge(df, country, on=['Location'], how='inner')
        fig = px.line(temp, x="Date", y="NewConfirmed", color="Location", title=f'Time Series for Top 10 countries with Confirmed cases on {recent_date.strftime("%m-%d-%Y")}')
        fig.update_layout(xaxis_rangeslider_visible=True)
             
    elif (gType == 'OverallCase'):
        #Display the timeseries data
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df2["Date"].tolist(), y=df2["NewConfirmed"].tolist(),
                            mode='lines',
                            name='New Confirmed'))
        fig.add_trace(go.Scatter(x=df2["Date"].tolist(), y=df2["NewRecovered"].tolist(),
                            mode='lines', line=dict(color='green'),
                            name='New Recovered'))
        fig.add_trace(go.Scatter(x=df2["Date"].tolist(), y=df2["NewDeaths"].tolist(),
                            mode='lines', line=dict(color='red'), name='New Deaths'))
        fig.update_layout(title=f'New Cases over time in {feature}', xaxis_rangeslider_visible=True)

    elif (gType == 'ActiveCase'):
        df2['Active'] = df2['Confirmed']- df2['Recovered']- df2['Deaths']
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df2["Date"].tolist(),
            y=df2["Confirmed"],
            name='Total Cases',
            marker_color='indianred'
        ))
        fig.add_trace(go.Bar(
            x=df2["Date"].tolist(),
            y=df2["Active"],
            name='Active Cases',
            marker_color='lightsalmon'
        ))
        
        fig.update_layout(barmode='group', xaxis_tickangle=-45, title=f'Total cases vs active case over time in {feature}')
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON
    
def plot_twitter_analysis():

    urls = []
    basepath = f"Flask App/"
    
    #polarity_Score
    dfT['Polarity_Score'] = dfT['text'].apply(lambda x : TextBlob(x).sentiment.polarity)

    #subjectivity score
    dfT['Subjectivity_Score'] = dfT['text'].apply(lambda x : TextBlob(x).sentiment.subjectivity)
    dfT['created_at'] = pd.to_datetime(dfT['created_at']).dt.date
    sns.lineplot(x="created_at", y="Polarity_Score", data=dfT)
    plt.title('Polarity_Score')
    plt.xticks(rotation=25)
    url = f'static/images/Polarity_Score_Line.png'
    plt.savefig(basepath + url)
    urls.append(url)
    plt.close()
    
    #Distribution Plot
    sns.distplot(dfT.Polarity_Score)
    plt.title('Polarity_Score Distribution')
    url = f'static/images/Polarity_Score_Distributuion.png'
    plt.savefig(basepath + url)
    urls.append(url)
    plt.close()
    
    #Create WordCloud
    Text = dfT['text']
    #Joining all text into one text variable
    All_Tweets = " ".join(dfT['text'])
    #Remove Stopwords
    All_Tweets = remove_stopwords(All_Tweets)
    #Remove "Covid", "Covid 19" & "Coronavirus" text because we would like to know words other than the subject
    All_Tweets = All_Tweets.replace('Covid', '')
    All_Tweets = All_Tweets.replace('covid', '')
    All_Tweets = All_Tweets.replace('covid-19', '')
    All_Tweets = All_Tweets.replace('wuhanvirus', '')
    All_Tweets = All_Tweets.replace('coronavirus', '')
    wordcloud = WordCloud(max_font_size=50, max_words=50, background_color="black").generate(All_Tweets)
    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Top 50 words for COVID-19 in Twitter')
    plt.axis("off")
    url = f'static/images/WordCloud.png'
    plt.savefig(basepath + url)
    urls.append(url)
    plt.close()
    
    return urls
    
def plot_predictions(x, y, future_forcast, pred, algo_name, color, feature):
   
    fig = go.Figure()    
    # Create and style traces
    fig.add_trace(go.Scatter(x=x.flatten().tolist() , y=y.flatten().tolist() , name='Confirmed Cases',
                             line = dict(color='royalblue' )))
    fig.add_trace(go.Scatter(x=future_forcast.flatten().tolist(), y=pred.tolist(), name=algo_name,
                             line = dict(color=color, dash='dash')))

    # Edit the layout
    fig.update_layout(title=f'# of Coronavirus Cases Over Time using {algo_name} in {feature}',
                       xaxis_title='Days',
                       yaxis_title='# of Cases')
                       
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return graphJSON

def create_prediction(feature):

    figs = []
    
    df2 = df[df['Location'] == feature]
    
    n = 5

    df2 = df2.drop(df2.tail(n).index)
    dates = df2.Date.unique()

    previous_days = np.array([i for i in range(len(dates))]).reshape(-1, 1)
    world_cases = np.array(df2['Confirmed']).reshape(-1, 1)

    future_forcast = np.array([i for i in range(len(dates)+ 10)]).reshape(-1, 1)
    adjusted_dates = future_forcast[:-10]

    future_forcast_dates = []
    for i in range(len(future_forcast)):
        future_forcast_dates.append((earliest_date + datetime.timedelta(days=i)).strftime('%Y-%m-%d'))
        
    X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(previous_days, world_cases, test_size=0.42, shuffle=False)

    # Prediction using SVM
    svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=4, C=0.2)
    svm_confirmed.fit(X_train_confirmed, y_train_confirmed.ravel())
    svm_pred = svm_confirmed.predict(future_forcast)

    fig1 = plot_predictions(adjusted_dates, world_cases, future_forcast, svm_pred, 'SVM_Predictions', 'purple',feature)
    figs.append(fig1)

    # Prediction using Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
    poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
    poly_future_forcast = poly.fit_transform(future_forcast)

    linear_model = LinearRegression(normalize=True, fit_intercept=False)
    linear_model.fit(poly_X_train_confirmed, y_train_confirmed.ravel())
    test_linear_pred = linear_model.predict(poly_X_test_confirmed)
    linear_pred = linear_model.predict(poly_future_forcast)

    fig2 = plot_predictions(adjusted_dates, world_cases, future_forcast, linear_pred, 'Polynomial_Regression_Predictions', 'orange',feature)       
    figs.append(fig2)

    # Prediction using MLP
    mlp_confirmed = MLPRegressor(solver='lbfgs', max_iter=500)
    mlp_confirmed.fit(X_train_confirmed, y_train_confirmed.ravel())
    mlp_pred = mlp_confirmed.predict(future_forcast)

    fig3 = plot_predictions(adjusted_dates,world_cases, future_forcast, mlp_pred, 'MLP_Predictions', 'green',feature)
    figs.append(fig3)

    return figs

@app.route('/country', methods=['GET', 'POST'])
def change_features():

    feature = request.args['selected']
    graphs = []
    figs = []
    graphs.append(json.loads(create_plot(feature,'OverallCase')))
    graphs.append(json.loads(create_plot(feature,'ActiveCase')))
    temp = create_prediction(feature)
    figs.append(json.loads(temp[0]))
    figs.append(json.loads(temp[1]))
    figs.append(json.loads(temp[2]))
    return jsonify({'graphs': graphs, 'figs': figs})

    
if __name__ == '__main__':
    app.run(debug=True)
