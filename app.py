import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash import dash_table

import pandas as pd
from ast import literal_eval
from tqdm.notebook import tqdm
tqdm.pandas()
from datetime import datetime
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from string import punctuation
import re
nltk.download('punkt')
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import base64


data_path = 'snacks_prep.csv'
print('Loading dataset at: ',datetime.now())
df = pd.read_csv(data_path, sep='\t', converters={'doc_entities': literal_eval, 'doc_keyphrases': literal_eval})
print('Loaded dataset at: ',datetime.now())

#df = pd.read_csv('practice.csv') 

df_shape = df.shape

#print(df.columns)
unique_doc_urls = df['doc_url'].nunique()

dict1 = {
        'id' : [],        
        'sentences' : [],
        'doc_sentiment' : [],
        'doc_date' : [],
        'doc_title' : [],
        'doc_url' : [],
        'doc_entities' : [],
        'doc_keyphrases' : [],
        'doc_publish_location' :[]
        }

for i1 in list(df['id'].unique()):
    temp_df = df[df['id'] == i1].reset_index().drop(['index'],1)
    try:
        dict1['id'].append(i1)
        dict1['doc_date'].append(temp_df['doc_date'][0])
        dict1['doc_title'].append(temp_df['doc_title'][0])
        dict1['doc_url'].append(temp_df['doc_url'][0])
        dict1['doc_entities'].append(temp_df['doc_entities'][0])
        dict1['doc_keyphrases'].append(temp_df['doc_keyphrases'][0])
        dict1['doc_publish_location'].append(temp_df['doc_publish_location'][0])
        dict1['doc_sentiment'].append(temp_df['doc_sentiment'][0])
        dict1['sentences'].append('.'.join(temp_df['sentence']))
    except:
        dict1['sentences'].append(temp_df['sentence'][0])

final_df = pd.DataFrame(dict1)
final_df_shape = final_df.shape

final_df[['Continent','Country']] = final_df.doc_publish_location.str.split(",",expand=True)

fig_continent = px.histogram(final_df, x="Continent")
fig_country = px.histogram(final_df, x="Country")


english_stopwords = stopwords.words('english')+list(punctuation)+['snack','food','year','time','make','product','need','find','made']

def preprocess(text):
    lemmatizer = nltk.WordNetLemmatizer().lemmatize
    text = re.sub('\W+', ' ', str(text))
    text = re.sub(r'[0-9]+', '', text.lower())
    tokenize_text = nltk.word_tokenize(text)
    stop_words_free = [i for i in tokenize_text if i not in english_stopwords and len(i) > 3]
    stop_words_free = list(set(stop_words_free))
    return(stop_words_free)


conditions = [
    (final_df['doc_sentiment'] > 0),
    (final_df['doc_sentiment'] == 0),
    (final_df['doc_sentiment'] < 0)
    ]
values = ['Positive', 'Neutral', 'Negative']
final_df['Sentiments_label'] = np.select(conditions, values)


final_df['doc_date'] = pd.to_datetime(final_df['doc_date'])
final_df['doc_date_month'] = pd.to_datetime(final_df['doc_date']).dt.month

final_df1 = final_df.copy()

for i in range(len(final_df1)):
    final_df1['doc_entities'][i]  = ', '.join([j+':'+k for i in final_df1['doc_entities'][i] for j,k in i.items()])

for i in range(len(final_df)):
    final_df1['doc_keyphrases'][i]  = ', '.join(final_df1['doc_keyphrases'][i])

final_df['sentences_tokenise'] = final_df['sentences'].apply(preprocess)
final_df['sentences'] = final_df['sentences'].str.lower()

colours = {
    "Negative": "red",
    "Neutral": "blue",
    "Positive": "green",
}


fig_continent_senti_l = px.histogram(final_df.groupby(['Continent','Sentiments_label']).size().to_frame('count').reset_index(), x="Continent", y='count',
                 color='Sentiments_label', barmode='group',color_discrete_map=colours,
                 height=600)


def create_wordcloud(data,word,season):
    text = ''
    for i in range(len(data)):
        text += ' '+' '.join(data['sentences_tokenise'][i])
    wordcloud = WordCloud().generate(text)
    # Display the generated image:
    plt.figure( figsize=(20,10) )
    #plt.title("Wordcloud for season : "+season)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.savefig('wordcloud_'+word+'.jpg')


create_wordcloud(final_df[final_df['doc_date_month'].isin([12])].reset_index().drop(['index'],1),'dec','Christmas')

encoded_wordcloud_dec = base64.b64encode(open('wordcloud_dec.jpg', 'rb').read())

create_wordcloud(final_df[final_df['doc_date_month'].isin([6,7,8])].reset_index().drop(['index'],1),'summer','Summer')

encoded_wordcloud_summer = base64.b64encode(open('wordcloud_summer.jpg', 'rb').read())

def get_postive_ne_neutral(list_months,word,season):
    dff = final_df[(final_df['doc_date_month'].isin(list_months)) & (final_df['sentences'].str.contains(word,na = False))].reset_index().drop(['index'],1)
    no_postivies = len(dff[dff['Sentiments_label']=='Positive'])
    no_negative = len(dff[dff['Sentiments_label']=='Negative'])
    no_neutral = len(dff[dff['Sentiments_label']=='Neutral'])
    fig = px.pie(values=[no_postivies,no_negative,no_neutral], names=['Positive Docs','Negative Docs','Neutral Docs'], title='Impact of '+word+' in '+season+'.')
    return fig

fig_crsip_s_pie = get_postive_ne_neutral([6,7,8],'crisp','Summer')
fig_nuts_s_pie = get_postive_ne_neutral([6,7,8],'popcorn','Chistmas')
fig_pretzel_c_pie = get_postive_ne_neutral([12],'pretzel','Chistmas')

def get_plots_country(list_months,word,season):
    dff = final_df[(final_df['doc_date_month'].isin(list_months)) & (final_df['sentences'].str.contains(word,na = False)) ].reset_index().drop(['index'],1)
    df_plot1 = dff.groupby(['Country','Sentiments_label']).size().to_frame('count').reset_index()
    fig = px.histogram(df_plot1, x="Country", y='count',
                 color='Sentiments_label', barmode='group',
                 height=400,
                 title='Sentiments for '+word+' over the globe in '+season+' season.',
                 color_discrete_map=colours)
    return fig 

fig_p = get_plots_country([12],'pretzel','Christmas')
fig_c  = get_plots_country([6,7,8],'crisp','Summer')
fig_n = get_plots_country([6,7,8],'popcorn','Christmas')

topic1 = base64.b64encode(open('topic1.jpg', 'rb').read())
topic2 = base64.b64encode(open('topic2.jpg', 'rb').read())
topic3 = base64.b64encode(open('topic3.jpg', 'rb').read())

print("Done Caluclation now Printing....................")

app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.H1('NLP Hackathon - Objecties',style = {'color':'#F0F0F0','font-family': "Times New Roman",
                'font-size': '30px','text-align':'center'}),
                html.H5('1. To Understand the preference of food by the people over the globe in the Summer and Christmas Season.',style = {'color':'#F0F0F0','font-family': "Times New Roman",
                'font-size': '15px','text-align':'center'}),
                html.H5('2. Understand the specific number of topics from the data Using Topic Modeling.',style = {'color':'#F0F0F0','font-family': "Times New Roman",
                'font-size': '15px','text-align':'center'})],
                style={"backgroundColor":'#2E4053','border':'2px white solid', 'borderRadius':13,'padding':5}#'width':220
        ),
        html.Div(
            children=[
                html.P('Rows Before Grouping',style = {'color':'#D5D8DC','font-family': "Times New Roman",'font-size': '15px','text-align':'center',
                'position': 'relative','left': '0px','bottom':''}),
                    
                html.H2(str(df_shape[0]), style = {'color':'#F0F0F0','font-family': "Times New Roman",'font-size': '15px','text-align':'center',
                'position': 'relative','left': '0px','bottom':''})],

                style={"backgroundColor":'#2E4053', 'borderRadius':13,'width':375,'height':80,
                'position': 'relative','left': '0px','display': 'inline-block'}
        ),
        html.Div(
            children=[
                html.P('Columns Before Grouping',style = {'color':'#D5D8DC','font-family': "Times New Roman",'font-size': '15px','text-align':'center',
                'position': 'relative','left': '0px','top': '10px'}),
                    
                html.H2(str(df_shape[1]), style = {'color':'#F0F0F0','font-family': "Times New Roman",'font-size': '15px','text-align':'center',
                'position': 'relative','top': '10px'})],

                style={"backgroundColor":'#2E4053', 'borderRadius':13,'width':375,'height':80,
                'position': 'relative','bottom': '155px','left':'380px','bottom': '94px'}
        ),
        html.Div(
            children=[
                html.P('Rows After Grouping',style = {'color':'#D5D8DC','font-family': "Times New Roman",'font-size': '15px','text-align':'center',
                'position': 'relative','left': '0px','top': '10px'}),

                html.H2(str(final_df_shape[0]), style = {'color':'#F0F0F0','font-family': "Times New Roman",'font-size': '15px','text-align':'center',
                'position': 'relative','top': '10px'})],

                style={"backgroundColor":'#2E4053', 'borderRadius':13,'width':375,'height':80,
                'position': 'relative','bottom': '190px','left':'760px'}
        ),
        html.Div(
            children=[
                html.P('Columns After Grouping',style = {'color':'#D5D8DC','font-family': "Times New Roman",'font-size': '15px','text-align':'center',
                'position': 'relative','left': '0px','top': '10px'}),

                html.H2(str(final_df_shape[1]), style = {'color':'#F0F0F0','font-family': "Times New Roman",'font-size': '15px','text-align':'center',
                'position': 'relative','top': '10px'})],

                style={"backgroundColor":'#2E4053', 'borderRadius':13,'width':365,'height':80,
                'position': 'relative','bottom': '283px','left':'1140px'}
        ),
        html.Div(
                html.H5('Data Table',style = {'color':'#F0F0F0','font-family': "Times New Roman",
                'font-size': '30px','text-align':'center'}),
                style={"backgroundColor":'#2E4053','borderRadius':13,
                'position': 'relative','left':'0px','bottom':'331px'}
        ),     
        html.Div(
            dash_table.DataTable(
                    style_data={
                            'lineHeight': '4px'
                        },
                    tooltip_data=[
                            {
                                column: {'value': str(value), 'type': 'markdown'}
                                for column, value in row.items()
                            } for row in final_df1.to_dict('records')
                        ],
                    page_action='none',
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(220, 220, 220)',
                        }
                    ],
                    style_header={
                        'backgroundColor': 'rgb(210, 210, 210)',
                        'color': 'black',
                        'fontWeight': 'bold'
                    },
                    style_table={'height': '300px', 'overflowY': 'auto'},
                    data=final_df1.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in final_df1.columns],
                    page_size=2,
                    style_cell={
                        'textAlign': 'left',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                        'maxWidth': 115
                    }
                ),
            style={'position': 'relative','bottom':'392px','display':'inline-block'}
        ),
        
        html.Div([
                html.Div(children='''
                    Bar graph of total count of label(Positive, Negative, Neutral) in different Continents.
                '''),
                dcc.Graph(
                    figure=fig_continent_senti_l
                ),  
            ], 
            className='row',
            style={'position': 'relative','left':'0px','bottom':'386px'}
        ),

        html.Div(children=[
                    html.Div(children='''
                    WordCloud for Summer and Christmas
                ''',style = {"backgroundColor":'#2E4053','borderRadius':13,'color':'#F0F0F0','font-family': "Times New Roman",'font-size': '30px','text-align':'center'}),
                    html.Img(src='data:image/jpg;base64,{}'.format(encoded_wordcloud_summer.decode()),
                        style = {"width":"700px","height":"500px",'display': 'inline-block'}),
                    html.Img(src='data:image/jpg;base64,{}'.format(encoded_wordcloud_dec.decode()),
                        style = {"width":"700px","height":"500px",'display': 'inline-block'})    
                    ],                       
                style={'position': 'relative','left':'0px','bottom':'350px','width': '100%','display': 'inline-block'}
        ),

        html.Div([
            dcc.Graph(figure = fig_crsip_s_pie, style={'display': 'inline-block','width': '500px'}),
            dcc.Graph(figure = fig_nuts_s_pie, style={'display': 'inline-block','width': '500px'}),
            dcc.Graph(figure = fig_pretzel_c_pie, style={'display': 'inline-block','width': '500px'})],
            style={'position': 'relative','left':'0px','bottom':'386px','width': '100%', 'display': 'inline-block'}
        ),

        html.Div([
            dcc.Graph(figure = fig_c, style={'display': 'inline-block','width': '500px'}),
            dcc.Graph(figure = fig_n, style={'display': 'inline-block','width': '500px'}),
            dcc.Graph(figure = fig_p, style={'display': 'inline-block','width': '500px'})],
            style={'position': 'relative','left':'0px','bottom':'386px','width': '100%', 'display': 'inline-block'}
        ),

        html.Div(children=[
                    html.Div(children='''
                    Topic Modeling NLP: We found Three Topics - Food Products, Business, Nutrition
                '''),
                    html.P(children='''
                    Topic 1 : Food Products
                    ''',style = {"backgroundColor":'#2E4053','borderRadius':13,'color':'#F0F0F0','font-family': "Times New Roman",'font-size': '30px','text-align':'center'}),
                    html.Img(src='data:image/jpg;base64,{}'.format(topic1.decode()),
                        style = {"width":"900px","height":"500px"}),
                    html.P(children='''
                    Topic 2 : Business
                    ''',style = {"backgroundColor":'#2E4053','borderRadius':13,'color':'#F0F0F0','font-family': "Times New Roman",'font-size': '30px','text-align':'center'}),
                    html.Img(src='data:image/jpg;base64,{}'.format(topic2.decode()),
                        style = {"width":"900px","height":"500px"}),
                    html.P(children='''
                    Topic 3 : Nutrition
                    ''',style = {"backgroundColor":'#2E4053','borderRadius':13,'color':'#F0F0F0','font-family': "Times New Roman",'font-size': '30px','text-align':'center'}),
                    html.Img(src='data:image/jpg;base64,{}'.format(topic3.decode()),
                        style = {"width":"900px","height":"500px"})
                    ],                       
                style={'position': 'relative','left':'0px','bottom':'350px','width': '100%'}
        ),


    ]
)

if __name__ == "__main__":
    app.run_server(debug=False)