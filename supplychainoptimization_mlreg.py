import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Dashboard creation and title

st.set_page_config(page_title="Supply Chain Optimizer", layout='wide')
st.title('Supply Chain Optimization Dashboard')
st.markdown('An implementation of machine learning to forecast supply chain performance based on randomly-generated data modeled with cyclicality')

#Data generation

def generate_data ():

    # output is a list of dates of the past 30 days
    #       today's date   minus   some no of days, x    formatted as Y-m-d      x starts at 30, minus 1, till 0
    dates = [(datetime.now() - timedelta(days = x)).strftime('%Y-%m-%d') for x in range (90, 0, -1)]

    warehouses = ['Warehouse A', 'Warehouse B', 'Warehouse C']
    centers = ['Center 1', 'Center 2', 'Center 3', 'Center 4', 'Center 5']

    data = []
    for date in dates:
        # Adding extra features
        dateobj = datetime.strptime(date,'%Y-%m-%d')
        day_of_week = dateobj.weekday() 
        day_of_month = dateobj.day
        month = dateobj.month

        # higher demand on mid-week and lower demand on weekends
        weekly_factor = 1.0
        if day_of_week >= 5:
            weekly_factor = 0.7
        elif day_of_week >= 1 and day_of_week <= 3:
            weekly_factor = 1.3

        # higher demand at start and end of month
        monthly_factor = 1.0
        if day_of_month <= 5 or day_of_month >=25:
            monthly_factor = 1.3

        # seasonality throughout year
        seasonal_factor = 1 + 0.3*(np.sin(month * np.pi / 6))

        for warehouse in warehouses:
            # variation between warehouses
            warehouse_factor = 1.0
            if warehouse == 'Warehouse A':
                warehouse_factor = 1.2
            elif warehouse == 'Warehouse B':
                warehouse_factor = 0.9
            
            for center in centers:
                # variation between centers
                center_factor = 1.0
                if center == 'Center 1':
                    center_factor = 1.1
                elif center == 'Center 2':
                    center_factor = 0.95

                # output is arbitrary seasonal value for base unit demand, will be used to model total unit demand
                # takes indexes of date in dates (1-30), takes the sin to model cyclicality, and scales to hover around a center of 100
                base_demand = (100 + 20 * np.sin(dates.index(date)) / 5)

                sim_variance = weekly_factor*monthly_factor*seasonal_factor*warehouse_factor*center_factor

                # output is randomly generated unit demand value with modeled market cyclicality
                #  marks lowest value as 0,   converts values to whole numbers,   base demand + random value centered at 0, std of 15  
                demand = (max(0, int(base_demand * sim_variance + np.random.normal(0, 15))))

                distance = np.random.randint(50, 500)
                cost_per_unit = round(0.1 + (distance * 0.01), 2)
                total_cost = round(demand * cost_per_unit, 2)

                # creates and inserts column in the DataFrame "data" as 'column title' : row value
                # between all the for loops, outputs 450 rows: 5 centers from each of the 3 warehouses for each of the 30 days
                data.append({
                    'date': date,
                    'day_of_week': day_of_week,
                    'day_of_month': day_of_month,
                    'month': month,
                    'warehouse': warehouse,
                    'distribution_center': center,
                    'units_shipped': demand,
                    'distance_miles': distance,
                    'cost_per_unit': cost_per_unit,
                    'total_cost': total_cost
                })
    
    return pd.DataFrame(data)

df = generate_data()

# Dashboard items + configuration

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

#Graphs

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

# Prepare data for advanced forecasting
forecast_df_daily = filtered_df.groupby(['date', 'day_of_week', 'day_of_month', 'month']).agg({
    'units_shipped': 'sum'
}).reset_index()

# Create feature engineering for time series
forecast_df_daily['day_num'] = range(len(forecast_df_daily))
forecast_df_daily['sin_month'] = np.sin(2 * np.pi * forecast_df_daily['month'] / 12)
forecast_df_daily['cos_month'] = np.cos(2 * np.pi * forecast_df_daily['month'] / 12)
forecast_df_daily['sin_day_of_week'] = np.sin(2 * np.pi * forecast_df_daily['day_of_week'] / 7)
forecast_df_daily['cos_day_of_week'] = np.cos(2 * np.pi * forecast_df_daily['day_of_week'] / 7)
forecast_df_daily['sin_day_of_month'] = np.sin(2 * np.pi * forecast_df_daily['day_of_month'] / 30)
forecast_df_daily['cos_day_of_month'] = np.cos(2 * np.pi * forecast_df_daily['day_of_month'] / 30)

# Create model tabs
model_tabs = st.tabs(["Simple Linear Regression", "Multiple Linear Regression"])

with model_tabs[0]:
    st.subheader("Simple Linear Regression Model")
    
    # Simple linear regression (original model)
    X_simple = forecast_df_daily[['day_num']]
    y = forecast_df_daily['units_shipped']
    
    model_simple = LinearRegression()
    model_simple.fit(X_simple, y)
    
    # Make future predictions with simple model
    future_days = 15
    future_day_nums = pd.DataFrame({'day_num': range(len(forecast_df_daily), len(forecast_df_daily) + future_days)})
    future_predictions_simple = model_simple.predict(future_day_nums)
    
    # Generate future dates
    last_date = datetime.strptime(forecast_df_daily['date'].iloc[-1], '%Y-%m-%d')
    future_dates = [(last_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, future_days + 1)]
    
    # Create dataframe for visualization
    forecast_simple_df = pd.DataFrame({
        'Date': list(forecast_df_daily['date']) + future_dates,
        'Units Shipped': list(forecast_df_daily['units_shipped']) + list(np.round(future_predictions_simple, 2)),
        'Data Source': ['Historical'] * len(forecast_df_daily) + ['Forecasted'] * len(future_dates)
    })
    
    # Plot simple regression forecast
    simple_forecast_fig = px.line(forecast_simple_df, x='Date', y='Units Shipped', color='Data Source', 
                                 title="Simple Linear Regression Forecast")
    st.plotly_chart(simple_forecast_fig, use_container_width=True)
    
    # Display model coefficients
    st.write(f"Simple Model Coefficient: {model_simple.coef_[0]:.4f}")
    st.write(f"Simple Model Intercept: {model_simple.intercept_:.4f}")

with model_tabs[1]:
    st.subheader("Multiple Linear Regression Model")
    
    # Feature selection for multiple regression
    feature_cols = st.multiselect(
        'Select Features for Multiple Regression',
        options=['day_num', 'day_of_week', 'day_of_month', 'month', 
                 'sin_month', 'cos_month', 'sin_day_of_week', 'cos_day_of_week',
                 'sin_day_of_month', 'cos_day_of_month'],
        default=['day_num', 'sin_month', 'cos_month', 'sin_day_of_week', 'cos_day_of_week']
    )
    
    if feature_cols:
        # Prepare features for multiple regression
        X_multi = forecast_df_daily[feature_cols]
        
        # Split for validation
        X_train, X_test, y_train, y_test = train_test_split(X_multi, y, test_size=0.2, random_state=42)
        
        # Train multiple regression model
        model_multi = LinearRegression()
        model_multi.fit(X_train, y_train)
        
        # Calculate scores
        train_score = model_multi.score(X_train, y_train)
        test_score = model_multi.score(X_test, y_test)
        
        st.write(f"Model R² (Train): {train_score:.4f}")
        st.write(f"Model R² (Test): {test_score:.4f}")
        
        # Generate future features
        future_features = pd.DataFrame()
        future_features['day_num'] = range(len(forecast_df_daily), len(forecast_df_daily) + future_days)
        
        # Generate time-based features for future dates
        for i, date in enumerate(future_dates):
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            future_features.loc[i, 'day_of_week'] = date_obj.weekday()
            future_features.loc[i, 'day_of_month'] = date_obj.day
            future_features.loc[i, 'month'] = date_obj.month
            
        # Calculate cyclical features
        future_features['sin_month'] = np.sin(2 * np.pi * future_features['month'] / 12)
        future_features['cos_month'] = np.cos(2 * np.pi * future_features['month'] / 12)
        future_features['sin_day_of_week'] = np.sin(2 * np.pi * future_features['day_of_week'] / 7)
        future_features['cos_day_of_week'] = np.cos(2 * np.pi * future_features['day_of_week'] / 7)
        future_features['sin_day_of_month'] = np.sin(2 * np.pi * future_features['day_of_month'] / 30)
        future_features['cos_day_of_month'] = np.cos(2 * np.pi * future_features['day_of_month'] / 30)
        
        # Make predictions using only selected features
        future_predictions_multi = model_multi.predict(future_features[feature_cols])
        
        # Create dataframe for visualization
        forecast_multi_df = pd.DataFrame({
            'Date': list(forecast_df_daily['date']) + future_dates,
            'Units Shipped': list(forecast_df_daily['units_shipped']) + list(np.round(future_predictions_multi, 2)),
            'Data Source': ['Historical'] * len(forecast_df_daily) + ['Forecasted'] * len(future_dates)
        })
        
        # Plot multiple regression forecast
        multi_forecast_fig = px.line(forecast_multi_df, x='Date', y='Units Shipped', color='Data Source', 
                                    title="Multiple Linear Regression Forecast")
        st.plotly_chart(multi_forecast_fig, use_container_width=True)
        
        # Display feature importance
        coefficients = pd.DataFrame({
            'Feature': feature_cols,
            'Coefficient': model_multi.coef_
        })
        coefficients = coefficients.sort_values('Coefficient', ascending=False)
        
        fig_coef = px.bar(coefficients, x='Feature', y='Coefficient', 
                         title='Feature Importance (Coefficients)')
        st.plotly_chart(fig_coef, use_container_width=True)
        
        # Compare both models
        comparison_df = pd.DataFrame({
            'Date': future_dates,
            'Simple Model': np.round(future_predictions_simple, 2),
            'Multiple Regression': np.round(future_predictions_multi, 2)
        })
        
        fig_compare = px.line(comparison_df, x='Date', y=['Simple Model', 'Multiple Regression'],
                             title='Forecast Comparison: Simple vs Multiple Regression')
        st.plotly_chart(fig_compare, use_container_width=True)
