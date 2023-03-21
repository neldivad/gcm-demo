import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import random
#-------------------------------------------------------------

def random_timeseries(initial_value: float, volatility: float, count: int) -> list:
  time_series = [initial_value,]
  for _ in range(count):
    time_series.append(time_series[-1] + initial_value * random.gauss(0.1,0.4) * volatility)
  return time_series
  
def gc_toy(shift=0):
  toy = pd.DataFrame()
  toy['data'] = ts
  toy[f'shift{shift}'] = toy['data'].shift(-shift)
  toy[f'shift{shift+2}'] = toy['data'].shift(-shift-2)
  
  mu=0.2
  std = 0.55
  def gaussian_noise(x,mu,std):
      noise = np.random.normal(mu, std, size = x.shape)
      x_noisy = x + noise
      return x_noisy
  toy[f'shift{shift}'] = gaussian_noise(toy[f'shift{shift}'], mu, std)
  toy[f'shift{shift+2}'] = gaussian_noise(toy[f'shift{shift+2}'], mu, std)
  
  toy = toy.dropna()
  dtoy = toy.pct_change().dropna() *100
  
  col1, col2, col3, col4, col5 = st.columns([0.2, 1, 0.5, 1, 0.2])
  with col2:
    lag_sel = st.slider('Select Lags', 1, 25, max(1,shift), key=10+shift)
  with col4:
    gcm_display = st.select_slider('Select Granger Matrix display', ['Only selected lag', 'Up to selected lags'], key=10+shift)
  if gcm_display == 'Only selected lag':
    all_lags= False
  else:
    all_lags= True

  st.markdown('#### Simulated time series plot')
  toy_plot = make_line_chart(toy, title= f'Simulated level data (shift={shift})', height=400, rangeslider=False)
  toy_gcm  = make_gc_matrix(toy, maxlag= lag_sel, all_lags=all_lags, title=f'P-values for Granger-causality for level data, lag= {lag_sel}')
  st.plotly_chart(toy_plot, use_container_width=True)
  st.plotly_chart(toy_gcm, use_container_width=True)
  
  st.markdown('#### Simulated time series (differenced) plot')
  dtoy_plot = make_line_chart(dtoy, title=f'Simulated differenced data (shift={shift})',height=400, rangeslider=False)
  dtoy_gcm  = make_gc_matrix(dtoy, maxlag= lag_sel, all_lags=all_lags, title=f'P-values for Granger-causality for differenced data, lag= {lag_sel}')
  st.plotly_chart(dtoy_plot, use_container_width=True)
  st.plotly_chart(dtoy_gcm, use_container_width=True)

def gc_example(shift= 0):
  data = sm.datasets.macrodata.load_pandas()
  level_con_gc_gdp = data.data[["realgdp", "realcons"]].dropna()
  dlevel_con_gc_gdp = data.data[["realgdp", "realcons"]].pct_change().dropna() *100
  level_gdp_gc_con = data.data[["realcons", "realgdp"]].dropna()
  dlevel_gdp_gc_con = data.data[["realcons", "realgdp"]].pct_change().dropna() *100

  level_con_gc_gdp['realcons']  = level_con_gc_gdp['realcons'].shift(-shift)
  dlevel_con_gc_gdp['realcons'] = dlevel_con_gc_gdp['realcons'].shift(-shift)
  level_gdp_gc_con['realcons']  = level_gdp_gc_con['realcons'].shift(-shift)
  dlevel_gdp_gc_con['realcons'] = dlevel_gdp_gc_con['realcons'].shift(-shift)

  level_con_gc_gdp  = level_con_gc_gdp.dropna()
  dlevel_con_gc_gdp = dlevel_con_gc_gdp.dropna()
  level_gdp_gc_con  = level_gdp_gc_con.dropna()
  dlevel_gdp_gc_con = dlevel_gdp_gc_con.dropna()
  
  alpha  = 0.05
  col1, col2, col3, col4, col5 = st.columns([0.2, 1, 0.5, 1, 0.2])
  with col2:
    lag_sel = st.slider('Select Lags', 1, 25, max(1,shift), key=shift)
  with col4:
    gcm_display = st.select_slider('Select Granger Matrix display', ['Only selected lag', 'Up to selected lags'], key=shift)
  if gcm_display == 'Only selected lag':
    all_lags= False
  else:
    all_lags= True
  
  gc_level_con_to_gdp  = grangercausalitytests(level_con_gc_gdp, [lag_sel], verbose= False)[lag_sel][0]['ssr_chi2test'][1].round(3)
  gc_dlevel_con_to_gdp = grangercausalitytests(dlevel_con_gc_gdp, [lag_sel], verbose=False)[lag_sel][0]['ssr_chi2test'][1].round(3)
  gc_level_gdp_to_con  = grangercausalitytests(level_gdp_gc_con, [lag_sel], verbose= False)[lag_sel][0]['ssr_chi2test'][1].round(3)
  gc_dlevel_gdp_to_con = grangercausalitytests(dlevel_gdp_gc_con, [lag_sel], verbose=False)[lag_sel][0]['ssr_chi2test'][1].round(3)
  
  level_line = make_line_chart(level_con_gc_gdp, title=f'Level data (shift={shift})', height=400, rangeslider=False)
  dlevel_line= make_line_chart(dlevel_con_gc_gdp, title=f'Differenced data (shift={shift})', height= 400, rangeslider=False)

  level_gcm  = make_gc_matrix(level_con_gc_gdp, maxlag= lag_sel, all_lags=all_lags, title=f'P-values for Granger-causality for level data, lag= {lag_sel}')
  dlevel_gcm = make_gc_matrix(dlevel_con_gc_gdp, maxlag= lag_sel, all_lags=all_lags, title=f'P-values for Granger-causality for differenced data, lag= {lag_sel}')

  st.markdown('#### Level data')
  st.plotly_chart(level_line, use_container_width=True)
  st.plotly_chart(level_gcm, use_container_width=True)
  if gc_level_con_to_gdp > alpha:
    st.write(f'P-value of `{gc_level_con_to_gdp}` is more than 0.05, meaning "realcons" **does not Granger-cause** "realgdp" when level data is used at lag of {lag_sel}.')
  else: 
    st.write(f'P-value of `{gc_level_con_to_gdp}` is less than 0.05, meaning "realcons" **does Granger-cause** "realgdp" when level data is used at lag of {lag_sel}.')
  if gc_level_gdp_to_con > alpha:
    st.write(f'P-value of `{gc_level_gdp_to_con}` is more than 0.05, meaning "realgdp" **does not Granger-cause** "realcons" when level data is used at lag of {lag_sel}.')
  else: 
    st.write(f'P-value of `{gc_level_gdp_to_con}` is less than 0.05, meaning "realgdp" **does Granger-cause** "realcons" when level data is used at lag of {lag_sel}.')

  st.markdown('#### Differenced data')
  st.plotly_chart(dlevel_line, use_container_width=True)
  st.plotly_chart(dlevel_gcm, use_container_width=True)
  if gc_dlevel_con_to_gdp > alpha:
    st.write(f'P-value of `{gc_dlevel_con_to_gdp}` is more than 0.05, meaning "realcons" **does not Granger-cause** "realgdp" when differenced data is used at lag of {lag_sel}.')
  else:
    st.write(f'P-value of `{gc_dlevel_con_to_gdp}` is less than 0.05, meaning "realcons" **does Granger-cause** "realgdp" when differenced data is used at lag of {lag_sel}.')
  if gc_dlevel_gdp_to_con > alpha:
    st.write(f'P-value of `{gc_dlevel_gdp_to_con}` is more than 0.05, meaning "realgdp" **does not Granger-cause** "realcons" when differenced data is used at lag of {lag_sel}.')
  else: 
    st.write(f'P-value of `{gc_dlevel_gdp_to_con}` is less than 0.05, meaning "realgdp" **does Granger-cause** "realcons" when differenced data is used at lag of {lag_sel}.')
    
#----------------    
def make_line_chart(df, title='', height=600, rangeslider=True):
  fig = px.line(
    df, x=df.index, y=df.columns.values
  )
  fig.update_traces(hovertemplate= '%{y}')
  fig.update_layout(
    title={
      'text': title,
      'y':1,
      'x':0.5,
      'xanchor': 'center',
      'yanchor': 'top'
    },
    legend=dict(
      x=0,
      y=1,
      title_text='',
      font=dict(
        family="Times New Roman",
        size=12,
        color="black"
      ),
      bgcolor= 'rgba(0,0,0,0)',
      bordercolor="Black",
      borderwidth=1
    ),
    hovermode="x unified",
    height= height,
  )
  fig.update_yaxes(title_text= '', autorange= True, fixedrange = False)
  fig.update_xaxes(
    title_text= '',
    rangeselector=dict(
      buttons=list([
        dict(count=1, label="1m", step="month", stepmode="backward"),
        dict(count=6, label="6m", step="month", stepmode="backward"),
        dict(count=1, label="YTD", step="year", stepmode="todate"),
        dict(count=1, label="1y", step="year", stepmode="backward"),
        dict(count=5, label="5y", step="year", stepmode="backward"),
        dict(count=10, label="10y", step="year", stepmode="backward"),
        dict(step="all")
        ])
    ),
    range= [ df.index.values.min(), df.index.values.max() ],
    rangeslider= dict(visible= rangeslider,),
  )
  return fig
  
def make_gc_matrix(dataframe, title, zmin=0, zmax=0.15, maxlag=12, all_lags=True, height=600, width=600):
  """
  Creates a granger causality matrix for a dataframe. Each cell contains a p-value. 
  A value of 0-0.10 means the null is rejected, and two variables have a G-cause relationship. 
  
  dataframe: Your input dataframe
  title: Your string title
  zmin: Default = 0. 
  zmax: Default = 0.15. Reject null at alpha 0.1. 
  maxlag: Default = 12. Lower lags will reduce the sensitivity of the test while higher lags increases it. 
          Too much lags produce spurious results. Use the lag order of VAR. 
  all_lags: When True, produce a GCM that returns the lowest P-value out of all lags tested
            by maxlag parameter. Else, produce a GCM that returns the P-value ONLY for lag
            of maxlag. 
  """
  def grangers_causality_matrix(data, variables, maxlag= 12, test = 'ssr_chi2test', verbose=False, all_lags=True):
    from statsmodels.tsa.stattools import grangercausalitytests

    dataset = pd.DataFrame(
      np.zeros((len(variables), len(variables))), 
      columns=variables, 
      index=variables)

    for c in dataset.columns:
      for r in dataset.index:
        if all_lags ==True:
          test_result = grangercausalitytests(
            data[[r,c]], 
            maxlag= maxlag, 
            verbose=False)
          p_values = [ test_result[i+1][0][test][1].round(4) for i in range(maxlag)]
        else:
          test_result = grangercausalitytests(
            data[[r,c]], 
            maxlag= [maxlag], 
            verbose=False)
          p_values = [ test_result[maxlag][0][test][1].round(4)]

        min_p_value = np.min(p_values)
        # min_p_value = p_values
        dataset.loc[r,c] = min_p_value

    dataset.columns = [var + '_x' for var in variables]
    dataset.index = [var + '_y' for var in variables]
    return dataset

  data = grangers_causality_matrix(dataframe, variables = dataframe.columns, maxlag= maxlag, all_lags=all_lags)  
  heat = go.Heatmap(
      z= data,
      x= data.columns.values,
      y= data.index.values,
      zmin= zmin,
      zmax= zmax,
      xgap= 1,
      ygap= 1,
      colorscale= 'BuPu',
      reversescale= True
  )
  layout = go.Layout(
      title_text=title, 
      title_x=0.5, 
      width= width, 
      height= height,
      xaxis_showgrid=False,
      yaxis_showgrid=False,
      yaxis_autorange='reversed',
  )
  fig= go.Figure(data=[heat], layout=layout)
  fig.add_annotation(
    text= '*Values in matrix are P-values for Granger causality',
    x= 0, 
    y= 1.075,
    xref="paper", 
    yref="paper",
    showarrow=False,
    font= dict(
      size= 12,
      color= 'black'
    )
  )
  return fig

#----------------------------------------------
st.subheader('Granger causality on economic data')
col1, col2, col3 = st.columns([2,10,2])
with col2:
  shif = st.selectbox('Time shift', [*range(1,10,1)], index=3, key='s1')
with st.expander(label='Granger causality economic data', expanded=False):
  gc_example(shift=0)
with st.expander(label=f'Granger causality economic data (shift={shif})', expanded=False):
  gc_example(shift=shif)

st.subheader('Granger causality on toy data')
col1, col2, col3, col4 = st.columns([2,10,10,2])
with col2:
  seed = st.selectbox('Seed number', [*range(1, 100, 1)], index=9)
with col3:
  shift = st.selectbox('Time shift', [*range(1,10,1)], index=3, key='s2')

random.seed(seed)
ts = random_timeseries(120, 0.01, 200)

with st.expander(label='Granger causality simulated data (shift=0)', expanded=False):
  gc_toy()
with st.expander(label=f'Granger causality simulated data (shift={shift})', expanded=False):
  gc_toy(shift= shift)
