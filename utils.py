import plotly.graph_objects as go


def create_gauge_chart(probability):
  if probability < 0.3:
    color = "green"
  elif probability < 0.6:
    color = "yellow"
  else:
    color = "red"

  fig = go.Figure(
      go.Indicator(mode="gauge+number",
                   value=probability * 100,
                   domain={
                       'x': [0, 1],
                       'y': [0, 1]
                   },
                   title={
                       'text': "Churn Probability",
                       'font': {
                           'size': 24,
                           'color': 'white'
                       }
                   },
                   number={'font': {
                       'size': 40,
                       'color': 'white'
                   }},
                   gauge={
                       'axis': {
                           'range': [0, 100],
                           'tickwidth': 1,
                           'tickcolor': 'white'
                       },
                       'bar': {
                           'color': color
                       },
                       'bgcolor':
                       "rgba(0,0,0,0)",
                       'borderwidth':
                       2,
                       'bordercolor':
                       "white",
                       'steps': [{
                           'range': [0, 30],
                           'color': "rgba(0,255,0,0.3)"
                       }, {
                           'range': [30, 60],
                           'color': "rgba(255,255,0,0.3)"
                       }, {
                           'range': [60, 100],
                           'color': "rgba(255,0,0,0.3)"
                       }],
                       'threshold': {
                           'line': {
                               'color': "white",
                               'width': 4
                           },
                           'thickness': 0.75,
                           'value': 100
                       }
                   }))

  fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': "white"},
                    width=400,
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20))
  return fig


def create_model_probability_chart(probabilities):
  models = list(probabilities.keys())
  probs = list(probabilities.values())

  fig = go.Figure(data=[
      go.Bar(y=models,
             x=probs,
             orientation='h',
             text=[f'{p:.2%}' for p in probs],
             textposition='auto')
  ])

  fig.update_layout(title="Churn Probability by Model",
                    yaxis_title="Models",
                    xaxis_title="Probability",
                    xaxis=dict(tickformat='.0%', range=[0, 1]),
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20))
  return fig
