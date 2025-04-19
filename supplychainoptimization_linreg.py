import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Supply Chain Optimizer", layout='wide')
st.title('Supply Chain Optimization Dashboard')
st.markdown('An implementation of machine learning to forecast supply chain performance based on randomly-generated data modeled with cyclicality')

def generate_data ():

    # output is a list of dates of the past 30 days
    #       today's date   minus   some no of days, x    formatted as Y-m-d      x starts at 30, minus 1, till 0
    dates = [(datetime.now() - timedelta(days = x)).strftime('%Y-%m-%d') for x in range (30, 0, -1)]

    warehouses = ['Warehouse A', 'Warehouse B', 'Warehouse C']
    centers = ['Center 1', 'Center 2', 'Center 3', 'Center 4', 'Center 5']

    data = []
    for date in dates:
        for warehouse in warehouses:
            for center in centers:
                # output is arbitrary seasonal value for base unit demand, will be used to model total unit demand
                # takes indexes of date in dates (1-30), takes the sin to model cyclicality, and scales to hover around a center of 100
                base_demand = (100 + 20 * np.sin(dates.index(date)) / 5)

                # output is randomly generated unit demand value with modeled market cyclicality
                #  marks lowest value as 0,   converts values to whole numbers,   base demand + random value centered at 0, std of 15  
                demand = (max(0, int(base_demand + np.random.normal(0, 15))))

                distance = np.random.randint(50, 500)
                cost_per_unit = round(0.1 + (distance * 0.01), 2)

                total_cost = round(demand * cost_per_unit, 2)

                # creates and inserts column in the DataFrame "data" as 'column title' : row value
                # between all the for loops, outputs 450 rows: 5 centers from each of the 3 warehouses for each of the 30 days
                data.append({
                    'date': date,
                    'warehouse': warehouse,
                    'distribution_center': center,
                    'units_shipped': demand,
                    'distance_miles': distance,
                    'cost_per_unit': cost_per_unit,
                    'total_cost': total_cost
                })
    
    return pd.DataFrame(data)

df = generate_data()

st.sidebar.header('Filters')

warehouse_filter = st.sidebar.multiselect(
    'Select Warehouses', 
    options=df['warehouse'].unique(), 
    default=df['warehouse'].unique()
)

center_filter = st.sidebar.multiselect(
    'Select Distribution Center', 
    options=df['distribution_center'].unique(), 
    default=df['distribution_center'].unique()
)


filtered_df = df[
    (df['warehouse'].isin(warehouse_filter)) &
    (df['distribution_center'].isin(center_filter))
]

st.header('Supply Chain Analysis')

col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Total Shipments', f"{filtered_df['units_shipped'].sum():,}")
with col2:
    st.metric("Total Cost", f"${filtered_df['total_cost'].sum():,.2f}")
with col3:
    st.metric("Avg Cost per Unit", f"${filtered_df['cost_per_unit'].mean():,.2f}")


#Creates a line graph plotting units shipped over time
# takes all the shipments for each date (groupby), sums them, and saves the dataframe as 'daily_shipments'
daily_shipments = filtered_df.groupby('date')['units_shipped'].sum().reset_index()
fig = px.line(daily_shipments, x='date', y='units_shipped', title='Daily Total Shipments', labels={
    'date': 'Date',
    'units_shipped': 'Units Shipped'
})
st.plotly_chart(fig, use_container_width=True)

#Creates bar graphs for various parameters for each warehouse
# pulls and groups the data by warehouse and sums up all units shipped and total costs -> df has 3 rows and 2 columns of data
warehouse_performance = filtered_df.groupby('warehouse').agg({
    'units_shipped' : 'sum',
    'total_cost' : 'sum'
    }).reset_index()

#creates a new column called CPU avg that takes total cost/total units for "cost per unit" 
warehouse_performance['cost_per_unit_avg'] = warehouse_performance['total_cost']/warehouse_performance['units_shipped']
fig2 = px.bar(warehouse_performance, x='warehouse', y='cost_per_unit_avg', title='Warehouse CPU Comparison', text_auto='.3s', labels={
    'warehouse': 'Warehouse',
    'cost_per_unit_avg': 'Average CPU'
})
fig2.update_yaxes(range = [2,3.5])
st.plotly_chart(fig2, use_container_width=False, horizontal=True)

fig3 = px.bar(warehouse_performance, x='warehouse', y='units_shipped',
            title='Total Units Shipped by Warehouse', text_auto='.3s', labels={
                'warehouse': 'Warehouse',
                'units_shipped': 'Units Shipped'
            })
fig3.update_yaxes(range=[14000,15500])
st.plotly_chart(fig3, use_container_width=True)

#Creates scatter plot displaying the cost and shipping distance ratios for each distribution center
center_performance = filtered_df.groupby('distribution_center').agg({
    'units_shipped': 'sum', #not doing anything with this currently but useful metric for other related plots
    'total_cost': 'sum',
    'distance_miles': 'mean'
}).reset_index()

fig4 = px.scatter(center_performance, x='distance_miles', y='total_cost', color='distribution_center', title='Distribution Center Analysis', labels={
    'distance_miles': 'Shipping Distance (mi)',
    'total_cost': 'Shipping Cost ($)'
})
fig4.update_traces(marker_size=10)
st.plotly_chart(fig4, use_container_width=True)

st.header("Demand Forecasting")
#Predictive Modeling

forecast_data = filtered_df.groupby('date').agg({'units_shipped': 'sum'}).reset_index()
forecast_data['day_num'] = range(len(forecast_data))


X = forecast_data[['day_num']]
Y = forecast_data['units_shipped']
model = LinearRegression()
model.fit(X, Y)

future_day_num = pd.DataFrame({'day_num': range(len(forecast_data), len(forecast_data) + 15)})
future_predictions = model.predict(future_day_num)

last_date = datetime.strptime(forecast_data['date'].iloc[-1], '%Y-%m-%d')
future_dates = [(last_date + timedelta(days = i)).strftime('%Y-%m-%d') for i in range(1,len(future_predictions) + 1)]

forecast_df = pd.DataFrame({
    'Dates': list(forecast_data['date']) + future_dates,
    'Units Shipped': list(forecast_data['units_shipped']) + list(np.round(future_predictions, 2)),
    'Data Source': ['Historical'] * len(forecast_data) + ['Forecasted'] * len(future_dates)
})

forecast_fig = px.line(forecast_df, x='Dates', y='Units Shipped', color='Data Source', title= datetime.now().strftime('%B') + " Shipping Performance with Future Forecast")
st.plotly_chart(forecast_fig, use_container_width=True)

