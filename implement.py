# -*- coding: utf-8 -*-
import dash
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#import plotly.plotly as py
import chart_studio.plotly as py
from plotly import tools
import plotly.express as px
import plotly.figure_factory as ff
import glob, os
import base64

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

product = "Amey Mohite"
Overall_Rating = 4.5
Product_Price  = 999.0
Reviews = 10000
Rating_reviews = 4.1
app = dash.Dash()

###############   Pie charts #################################
labels = ["Positive", "Neutral", "Negative"]

# Create subplots: use 'domain' type for Pie subplot
colors = ['gold', 'mediumturquoise', 'darkorange']

fig_pie = make_subplots(rows=1, cols=1, specs=[[{'type':'domain'}]])
fig_pie.add_trace(go.Pie(labels=labels, values=[16, 15, 12], name="GHG Emissions"),
              1, 1)
fig_pie.update_traces(hole=.4, hoverinfo="label+percent+name")

fig_pie.update_layout(
    title_text="Postive_Negative_Neutral",
    autosize=False,
    width=550,
    height=400,
    margin=go.layout.Margin(l=0, r=0,b=0,t=40, pad=0),
    annotations=[dict(text='Sentiment', x=0.55, y=0.5, font_size=20, showarrow=False)])


#######################   Bar  Chart ############################################
animals=['1', '2', '3','4','5']

fig_bar = go.Figure(data=[
    go.Bar(name='Customer Rating', x=animals, y=[20, 14, 23,54,212],marker_color='rgb(158,202,225)'),
    go.Bar(name='Review Rating', x=animals, y=[12, 18, 29,54,242],marker_color='rgb(250, 215, 160)')
])
fig_bar.update_traces(marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
# Change the bar mode
fig_bar.update_layout(barmode='group')

fig_bar.update_layout(
    title_text="Count of Stars",
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Rating",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    ),
    autosize=False,
    width=600,
    height=400,
    margin=go.layout.Margin(l=0, r=0,b=0,t=40, pad=0))

################################## table #########################################
values = [['<a href="https://plot.ly/~empet/folder/home">Lorem ipsum dolor sit amet, tollit discere inermis pri ut. Eos ea iusto timeam, an prima laboramus vim. Id usu aeterno adversarium, summo mollis timeam vel ad</a>',
                '<a href="https://plot.ly/python/">Lorem ipsum dolor sit amet, tollit discere inermis pri ut. Eos ea iusto timeam, an prima laboramus vim. Id usu aeterno adversarium, summo mollis timeam vel ad</a>',
               '<a href="https://plot.ly/~Grondo/folder/home">Lorem ipsum dolor sit amet, tollit discere inermis pri ut. Eos ea iusto timeam, an prima laboramus vim. Id usu aeterno adversarium, summo mollis timeam vel ad</a>',
                '<a href="https://plot.ly/matlab/">Lorem ipsum dolor sit amet, tollit discere inermis pri ut. Eos ea iusto timeam, an prima laboramus vim. Id usu aeterno adversarium, summo mollis timeam vel ad</a>',
               '<a href="https://plot.ly/~Dreamshot/folder/home">Lorem ipsum dolor sit amet, tollit discere inermis pri ut. Eos ea iusto timeam, an prima laboramus vim. Id usu aeterno adversarium, summo mollis timeam vel ad</a>'],
                ['<a href="https://help.plot.ly/tutorials/">Lorem ipsum dolor sit amet, tollit discere inermis pri ut. Eos ea iusto timeam, an prima laboramus vim. Id usu aeterno adversarium, summo mollis timeam vel ad</a>',
               '<a href="https://plot.ly/~FiveThirtyEight/folder/home">Lorem ipsum dolor sit amet, tollit discere inermis pri ut. Eos ea iusto timeam, an prima laboramus vim. Id usu aeterno adversarium, summo mollis timeam vel ad</a>',
                '<a href="https://help.plot.ly/tutorials/">Lorem ipsum dolor sit amet, tollit discere inermis pri ut. Eos ea iusto timeam, an prima laboramus vim. Id usu aeterno adversarium, summo mollis timeam vel ad</a>',
               '<a href="https://plot.ly/~cpsievert/folder/home">Lorem ipsum dolor sit amet, tollit discere inermis pri ut. Eos ea iusto timeam, an prima laboramus vim. Id usu aeterno adversarium, summo mollis timeam vel ad</a>',
                '<a href="https://plot.ly/r/">RLorem ipsum dolor sit amet, tollit discere inermis pri ut. Eos ea iusto timeam, an prima laboramus vim. Id usu aeterno adversarium, summo mollis timeam vel ad</a>']]

fig_table = go.Figure(data=[go.Table(
  columnorder = [1,2],
  columnwidth = [700,700],
  header = dict(
    values = [['<b>Top Positive Reviews</b>'],
                  ['<b>Top Negative Reviews</b>']],
    line_color='darkslategray',
    fill_color='#2ECC71' ,
    align=['center','center'],
    font=dict(color='white', size=12),
    height=40
  ),
  cells=dict(
    values=values,
    line_color='darkslategray',
    fill=dict(color=['#212F3D', '#212F3D']),
    align=['left', 'left'],
    font_size=12,
    height=30)
    )
])
fig_table.update_layout(
    title_text="Reviews",
    autosize=False,
    width=850,
    height=700,
    margin=go.layout.Margin(l=0, r=0,b=0,t=40, pad=0))

#################################################  Image Slideerr ###############################################


list_images = []
for file in glob.glob("*.jpg"):
    encoded_image = base64.b64encode(open(file, 'rb').read())
    list_images.append(encoded_image)

######################## First Positive plot Bar #####################################################
positive_labels = ["0-0.3", "0.3-0.6", "0.6-1"]

# Create subplots: use 'domain' type for Pie subplot
fig_bar_positive = go.Figure(data=[
    go.Bar(name='Positive Rating', x=positive_labels, y=[20, 14, 23],
            hovertext=['Review-Polarity : 0 to 0.3', 'Review-Polarity : 0.3 to 0.6', 'Review-Polarity : 0.6 to 1'])
])
fig_bar_positive.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
fig_bar_positive.update_layout(
    title_text="Positive Rating",
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Positive review Polarity Range",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Count of reviews",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    ),
    autosize=False,
    width=400,
    height=400,
    margin=go.layout.Margin(l=0, r=0,b=0,t=40, pad=0))

###############################################  Second negative plotss bar ##################################
negative_labels = ["0-(-0.3)", "(-0.3)-(-0.6)", "(-0.6)-(-1)"]
fig_bar_negative = go.Figure(data=[
    go.Bar(name='Negative Rating', x=negative_labels, y=[20, 14, 23],
            hovertext=['Review-Polarity : 0 to -0.3', 'Review-Polarity : -0.3 to -0.6', 'Review-Polarity : -0.6 to -1'])
])
fig_bar_negative.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
fig_bar_negative.update_layout(
    title_text="Negative Rating",
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Negative review Polarity Range",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Count of reviews",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    ),
    autosize=False,
    width=400,
    height=400,
    margin=go.layout.Margin(l=0, r=0,b=0,t=40, pad=0))

###############################################  word cloud  ########################################## 
encoded_wordcloud = base64.b64encode(open('wordcloud.png', 'rb').read())

#####################################################  frequency of words ##################################################
encoded_common = base64.b64encode(open('common_words_graph.png', 'rb').read())
###############################  End #################################################

app.layout =html.Div(children=[
##################################  Heading #################################################
html.Div(
html.H1('Sentiment Analysis for the '+product,style = {'color':'#F0F0F0','font-family': "Times New Roman",
'font-size': '30px','text-align':'center'}),
style={"backgroundColor":'#2E4053','border':'2px white solid', 'borderRadius':13,'padding':5}#'width':220
),
##############################   Rating  ######################################################    
html.Div(children=[
html.P('Rating',style = {'color':'#D5D8DC','font-family': "Times New Roman",'font-size': '25px','text-align':'center',
'position': 'relative','left': '0px'}),
    

html.H2(str(Overall_Rating), style = {'color':'#F0F0F0','font-family': "Times New Roman",'font-size': '30px','text-align':'center',
'position': 'relative','left': '0px'})],

style={"backgroundColor":'#2E4053','border':'2px white solid', 'borderRadius':13,'width':375,'height':150,
'position': 'relative','left': '0px','display': 'inline-block'}
),
#############################   Product Price ####################################################
html.Div(children=[
html.P('Price',style = {'color':'#D5D8DC','font-family': "Times New Roman",'font-size': '25px','text-align':'center',
'position': 'relative','bottom': '0px'}),
    
html.H2(str(Product_Price), style = {'color':'#F0F0F0','font-family': "Times New Roman",'font-size': '30px','text-align':'center',
'position': 'relative','bottom': '0px'})],

style={"backgroundColor":'#2E4053','border':'2px white solid', 'borderRadius':13,'width':375,'height':150,
'position': 'relative','bottom': '155px','left':'380px'}
),      
##############################  No. oF Reviews ###################################################
html.Div(children=[

html.P('Reviews',style = {'color':'#D5D8DC','font-family': "Times New Roman",'font-size': '25px','text-align':'center',
'position': 'relative','bottom': '0px'}),

html.H2(str(Reviews), style = {'color':'#F0F0F0','font-family': "Times New Roman",'font-size': '30px','text-align':'center',
'position': 'relative','bottom': '0px'})],

style={"backgroundColor":'#2E4053','border':'2px white solid', 'borderRadius':13,'width':375,'height':150,
'position': 'relative','bottom': '308px','left':'760px'}
),
##############################  Rating Based on reviews ###################################################
html.Div(children=[

html.P('Review Rating',style = {'color':'#D5D8DC','font-family': "Times New Roman",'font-size': '25px','text-align':'center',
'position': 'relative','bottom': '0px'}),

html.H2(str(Rating_reviews), style = {'color':'#F0F0F0','font-family': "Times New Roman",'font-size': '30px','text-align':'center',
'position': 'relative','bottom': '0px'})],

style={"backgroundColor":'#2E4053','border':'2px white solid', 'borderRadius':13,'width':365,'height':150,
'position': 'relative','bottom': '462px','left':'1140px'}
),
###############################################  Pie Chart of No of positive negavite reviews #############################################
html.Div(children=[
dcc.Graph(figure=fig_pie, id='my-figure_pie')
],
style={'position': 'relative','bottom': '460px','display':'inline-block'}##add style to the div tag,'width':375,'height':150 ,'left':'1140px'
),
################################################ First Positive Bar plotss  ##################################################
html.Div(children=[
dcc.Graph(figure=fig_bar_positive, id='my-figure_positive')
],
style={'position': 'relative','bottom': '460px','left':'70px','display': 'inline-block'}##add style to the div tag,'width':375,'height':150 ,'left':'1140px'
),
################################################ Second Negative Bar plotss  ##################################################
html.Div(children=[
dcc.Graph(figure=fig_bar_negative, id='my-figure_negative')
],
style={'position': 'relative','bottom': '460px','left':'110px','display': 'inline-block'}##add style to the div tag,'width':375,'height':150 ,'left':'1140px'
),
#############################################     Bar Graphs ###################################
html.Div(children=[
dcc.Graph(figure=fig_bar, id='my-figure_bar')
],
style={'position': 'relative','bottom': '720px','display': 'inline-block'}##add style to the div tag,'width':375,'height':150 ,'left':'1140px'
),
#############################################   tables   ###################################
html.Div(children=[
dcc.Graph(figure=fig_table, id='my-figure_table')
],
style={'position': 'relative','bottom': '420px','left':'30px','display': 'inline-block'}##add style to the div tag,'width':375,'height':150 ,'left':'1140px'
),
#####################################################  Image slider ##################################################
html.Div(children=[
dcc.Slider(id='my-slider', min=0,max=9,step=1,value=0),
html.Div(id='slider-output-container',
style={'display': 'inline-block'}
)
],
style = {'width':'600px','height':'500px','position': 'relative','bottom': '680px','left':'0px','display': 'inline-block'}
),
#####################################################  Word Cloud ##################################################
html.Div(children=[
                html.Img(src='data:image/jpg;base64,{}'.format(encoded_wordcloud.decode()),
                    style = {"width":"900px","height":"500px"})
                    ],                       
style={"backgroundColor":'#2E4053','width':'600px','height':'500px','position': 'relative','bottom': '680px','display': 'inline-block'}
),
#####################################################  frequency of words ##################################################
html.Div(children=[
                html.Img(src='data:image/jpg;base64,{}'.format(encoded_common.decode()),
                    style = {"width":"1000","height":"500px"})
                    ],                       
style={'width':'1000px','height':'500px','position': 'relative','bottom': '680px','display': 'inline-block'}
)
],
            style = {'display': 'inline-block','margin': '0px'}#"backgroundColor":'#FFF0F5'
)
@app.callback(
    dash.dependencies.Output('slider-output-container', 'children'),
    [dash.dependencies.Input('my-slider', 'value')])
def update_output(value): 
    return html.Img(src='data:image/jpg;base64,{}'.format(list_images[value].decode()),
                    style = {"width":"600px","height":"500px"})


if __name__ == '__main__':
    app.run_server()