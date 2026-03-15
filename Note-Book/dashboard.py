"""
Price Forecasting and Prediction Dashboard
A Streamlit dashboard for analyzing and forecasting commodity prices
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Price Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None

@st.cache_data
def load_data():
    """
    Load and prepare the price data
    In production, replace with actual data loading from your source
    """
    # This is a placeholder - replace with your actual data loading logic
    # Example: df = pd.read_csv('your_data.csv')
    
    # For demonstration, creating sample data structure based on your info
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='15D')
    
    data = {
        'date': np.random.choice(dates, 1000),
        'admin1': np.random.choice(['Addis Ababa', 'Amhara', 'Oromia', 'Tigray', 'SNNPR'], 1000),
        'admin2': np.random.choice(['Zone1', 'Zone2', 'Zone3', 'Zone4'], 1000),
        'market': np.random.choice(['Market A', 'Market B', 'Market C', 'Market D'], 1000),
        'latitude': np.random.uniform(3.0, 15.0, 1000),
        'longitude': np.random.uniform(33.0, 48.0, 1000),
        'category': np.random.choice(['cereals and tubers', 'pulses and nuts', 'oil and fats', 
                                     'meat, fish and eggs', 'non-food'], 1000),
        'commodity': np.random.choice(['Maize', 'Sorghum', 'Teff', 'Wheat', 'Lentils'], 1000),
        'unit': np.random.choice(['100 KG', 'KG', 'L', 'HEAD', 'DAY'], 1000),
        'priceflag': np.random.choice(['Actual', 'Estimated'], 1000),
        'pricetype': np.random.choice(['Retail', 'Wholesale'], 1000),
        'currency': 'ETB',
        'price': np.random.uniform(50, 5000, 1000),
        'usdprice': np.random.uniform(1, 100, 1000),
        'year': 0,
        'month': 0,
        'commodity_base': np.random.choice(['Maize', 'Sorghum', 'Teff', 'Wheat', 'Lentils'], 1000),
        'variety': np.random.choice(['white', 'red', 'mixed', 'standard'], 1000)
    }
    
    df = pd.DataFrame(data)
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month'] = pd.DatetimeIndex(df['date']).month
    
    return df

def create_time_series_plot(data, commodity, market):
    """Create interactive time series plot"""
    fig = go.Figure()
    
    # Filter data for specific commodity and market
    plot_data = data[(data['commodity_base'] == commodity) & (data['market'] == market)]
    
    if not plot_data.empty:
        fig.add_trace(go.Scatter(
            x=plot_data['date'],
            y=plot_data['price'],
            mode='lines+markers',
            name=f'{commodity} - {market}',
            line=dict(color='#1E88E5', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=f'Price Trends: {commodity} at {market}',
            xaxis_title='Date',
            yaxis_title='Price (ETB)',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
    return fig

def create_heatmap(data):
    """Create correlation heatmap"""
    # Select numeric columns for correlation
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr_matrix = data[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Feature Correlation Matrix"
    )
    
    fig.update_layout(height=500)
    return fig

def train_forecast_model(data, commodity, features, target='price'):
    """Train a simple forecasting model"""
    # Prepare data
    commodity_data = data[data['commodity_base'] == commodity].copy()
    
    if len(commodity_data) < 10:
        return None, None, None
    
    # Create time-based features
    commodity_data['dayofyear'] = commodity_data['date'].dt.dayofyear
    commodity_data['quarter'] = commodity_data['date'].dt.quarter
    
    # Select features
    feature_cols = ['year', 'month', 'dayofyear', 'quarter'] + features
    
    # Ensure all features exist
    feature_cols = [f for f in feature_cols if f in commodity_data.columns]
    
    if len(feature_cols) < 2:
        return None, None, None
    
    X = commodity_data[feature_cols].fillna(0)
    y = commodity_data[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, y_test, y_pred, mse, r2

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<div class="main-header">📊 Price Forecasting & Prediction Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1E88E5/ffffff?text=Price+Analytics", 
                 use_column_width=True)
        st.markdown("## 📋 Controls")
        
        # Data source selection
        data_source = st.radio(
            "Select Data Source",
            ["Sample Data", "Upload Your Data"],
            help="Choose between sample data or upload your own"
        )
        
        if data_source == "Upload Your Data":
            uploaded_file = st.file_uploader(
                "Upload CSV file", 
                type=['csv'],
                help="Upload your price data CSV file"
            )
            if uploaded_file is not None:
                try:
                    st.session_state.data = pd.read_csv(uploaded_file)
                    st.success("✅ Data loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading data: {e}")
            else:
                st.info("Please upload a CSV file")
        else:
            if st.button("🔄 Load Sample Data"):
                with st.spinner("Loading data..."):
                    st.session_state.data = load_data()
                st.success("✅ Sample data loaded!")
        
        st.markdown("---")
        
        # Filters (if data is loaded)
        if st.session_state.data is not None:
            st.markdown("## 🔍 Filters")
            
            # Create filters
            commodities = ['All'] + sorted(st.session_state.data['commodity_base'].unique().tolist())
            selected_commodity = st.selectbox("Commodity", commodities)
            
            markets = ['All'] + sorted(st.session_state.data['market'].unique().tolist())
            selected_market = st.selectbox("Market", markets)
            
            date_range = st.date_input(
                "Date Range",
                value=(
                    st.session_state.data['date'].min(),
                    st.session_state.data['date'].max()
                ),
                min_value=st.session_state.data['date'].min(),
                max_value=st.session_state.data['date'].max()
            )
            
            # Apply filters
            filtered = st.session_state.data.copy()
            
            if selected_commodity != 'All':
                filtered = filtered[filtered['commodity_base'] == selected_commodity]
            
            if selected_market != 'All':
                filtered = filtered[filtered['market'] == selected_market]
            
            if len(date_range) == 2:
                filtered = filtered[
                    (filtered['date'] >= pd.to_datetime(date_range[0])) &
                    (filtered['date'] <= pd.to_datetime(date_range[1]))
                ]
            
            st.session_state.filtered_data = filtered
            
            st.markdown(f"**📊 Records:** {len(filtered):,}")
    
    # Main content area
    if st.session_state.data is None:
        # Welcome message
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📈 Price Forecasting</h3>
                <p>Predict future prices using machine learning models</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Market Analysis</h3>
                <p>Analyze price trends across different markets</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🤖 Price Classification</h3>
                <p>Classify price movements and patterns</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("👈 Please load data from the sidebar to get started!")
        
    else:
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Overview", 
            "📈 Price Trends", 
            "🔮 Forecasting",
            "📉 Price Classification",
            "🗺️ Market Analysis"
        ])
        
        # Tab 1: Data Overview
        with tab1:
            st.header("📊 Data Overview")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Records",
                    f"{len(st.session_state.data):,}",
                    delta=None
                )
            
            with col2:
                avg_price = st.session_state.data['price'].mean()
                st.metric(
                    "Average Price",
                    f"{avg_price:,.2f} ETB"
                )
            
            with col3:
                unique_markets = st.session_state.data['market'].nunique()
                st.metric(
                    "Markets",
                    f"{unique_markets:,}"
                )
            
            with col4:
                date_range = (st.session_state.data['date'].max() - 
                            st.session_state.data['date'].min()).days
                st.metric(
                    "Date Range",
                    f"{date_range} days"
                )
            
            # Data table
            st.subheader("📋 Data Sample")
            st.dataframe(
                st.session_state.data.head(10),
                use_container_width=True,
                hide_index=True
            )
            
            # Data info
            with st.expander("📊 Data Information"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Data Types:**")
                    dtypes_df = pd.DataFrame({
                        'Column': st.session_state.data.dtypes.index,
                        'Data Type': st.session_state.data.dtypes.values.astype(str)
                    })
                    st.dataframe(dtypes_df, hide_index=True)
                
                with col2:
                    st.write("**Missing Values:**")
                    missing_df = pd.DataFrame({
                        'Column': st.session_state.data.isnull().sum().index,
                        'Missing': st.session_state.data.isnull().sum().values,
                        'Percentage': (st.session_state.data.isnull().sum().values / 
                                     len(st.session_state.data) * 100).round(2)
                    })
                    st.dataframe(missing_df[missing_df['Missing'] > 0], hide_index=True)
        
        # Tab 2: Price Trends
        with tab2:
            st.header("📈 Price Trends Analysis")
            
            if st.session_state.filtered_data is not None and len(st.session_state.filtered_data) > 0:
                # Commodity selector for trends
                selected_commodity_trend = st.selectbox(
                    "Select Commodity for Trend Analysis",
                    options=sorted(st.session_state.data['commodity_base'].unique()),
                    key="trend_commodity"
                )
                
                selected_market_trend = st.selectbox(
                    "Select Market",
                    options=sorted(st.session_state.data['market'].unique()),
                    key="trend_market"
                )
                
                # Create time series plot
                fig = create_time_series_plot(
                    st.session_state.data, 
                    selected_commodity_trend, 
                    selected_market_trend
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics for selected combination
                trend_data = st.session_state.data[
                    (st.session_state.data['commodity_base'] == selected_commodity_trend) &
                    (st.session_state.data['market'] == selected_market_trend)
                ]
                
                if not trend_data.empty:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Min Price",
                            f"{trend_data['price'].min():,.2f} ETB"
                        )
                    
                    with col2:
                        st.metric(
                            "Max Price",
                            f"{trend_data['price'].max():,.2f} ETB"
                        )
                    
                    with col3:
                        st.metric(
                            "Average Price",
                            f"{trend_data['price'].mean():,.2f} ETB"
                        )
            else:
                st.warning("No data available for the selected filters")
        
        # Tab 3: Forecasting
        with tab3:
            st.header("🔮 Price Forecasting")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                forecast_commodity = st.selectbox(
                    "Select Commodity for Forecasting",
                    options=sorted(st.session_state.data['commodity_base'].unique()),
                    key="forecast_commodity"
                )
            
            with col2:
                # Feature selection for forecasting
                st.write("**Select Features for Model:**")
                use_latlong = st.checkbox("Use Latitude/Longitude", value=False)
                use_category = st.checkbox("Use Category", value=False)
                use_unit = st.checkbox("Use Unit", value=False)
            
            if st.button("🚀 Train Forecasting Model", type="primary"):
                with st.spinner("Training model..."):
                    # Prepare features
                    features = []
                    if use_latlong:
                        features.extend(['latitude', 'longitude'])
                    
                    # Train model
                    result = train_forecast_model(
                        st.session_state.data, 
                        forecast_commodity,
                        features
                    )
                    
                    if result[0] is not None:
                        model, y_test, y_pred, mse, r2 = result
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Model R² Score", f"{r2:.3f}")
                        
                        with col2:
                            st.metric("Mean Squared Error", f"{mse:.2f}")
                        
                        with col3:
                            st.metric("RMSE", f"{np.sqrt(mse):.2f}")
                        
                        # Plot actual vs predicted
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=list(range(len(y_test))),
                            y=y_test.values,
                            mode='lines+markers',
                            name='Actual',
                            line=dict(color='#1E88E5')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=list(range(len(y_pred))),
                            y=y_pred,
                            mode='lines+markers',
                            name='Predicted',
                            line=dict(color='#FFA000', dash='dash')
                        ))
                        
                        fig.update_layout(
                            title=f'Actual vs Predicted Prices - {forecast_commodity}',
                            xaxis_title='Sample',
                            yaxis_title='Price (ETB)',
                            template='plotly_white',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Feature importance
                        if hasattr(model, 'feature_importances_'):
                            importance_df = pd.DataFrame({
                                'Feature': features + ['year', 'month', 'dayofyear', 'quarter'],
                                'Importance': model.feature_importances_
                            }).sort_values('Importance', ascending=True)
                            
                            fig_importance = px.bar(
                                importance_df,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title='Feature Importance',
                                color='Importance',
                                color_continuous_scale='Blues'
                            )
                            
                            st.plotly_chart(fig_importance, use_container_width=True)
                    else:
                        st.error("Insufficient data for training. Please select a different commodity.")
        
        # Tab 4: Price Classification
        with tab4:
            st.header("📉 Price Movement Classification")
            
            st.info("""
            This section classifies price movements into categories:
            - **Up**: Price increased
            - **Down**: Price decreased
            - **Stable**: Price remained relatively constant
            """)
            
            # Prepare classification data
            if st.session_state.filtered_data is not None and len(st.session_state.filtered_data) > 0:
                # Calculate price changes
                classification_data = st.session_state.filtered_data.copy()
                classification_data = classification_data.sort_values(['commodity_base', 'market', 'date'])
                
                # Group by commodity and market to calculate price changes
                classification_data['price_change'] = classification_data.groupby(
                    ['commodity_base', 'market']
                )['price'].pct_change() * 100
                
                # Create categories
                classification_data['movement'] = pd.cut(
                    classification_data['price_change'],
                    bins=[-float('inf'), -5, 5, float('inf')],
                    labels=['Down', 'Stable', 'Up']
                )
                
                # Plot distribution
                movement_counts = classification_data['movement'].value_counts()
                
                fig = px.pie(
                    values=movement_counts.values,
                    names=movement_counts.index,
                    title="Price Movement Distribution",
                    color_discrete_sequence=['#ef5350', '#ffa000', '#4caf50']
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show sample
                st.subheader("Sample Price Movements")
                st.dataframe(
                    classification_data[['date', 'commodity_base', 'market', 'price', 'price_change', 'movement']]
                    .dropna()
                    .head(10),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("Please apply filters to view classification data")
        
        # Tab 5: Market Analysis
        with tab5:
            st.header("🗺️ Market Analysis")
            
            if st.session_state.filtered_data is not None and len(st.session_state.filtered_data) > 0:
                # Geographic distribution
                st.subheader("Market Locations")
                
                fig_map = px.scatter_mapbox(
                    st.session_state.filtered_data,
                    lat='latitude',
                    lon='longitude',
                    color='price',
                    size='price',
                    hover_name='market',
                    hover_data=['commodity_base', 'price', 'date'],
                    color_continuous_scale='Viridis',
                    zoom=5,
                    mapbox_style='open-street-map',
                    title='Price Distribution Across Markets'
                )
                
                fig_map.update_layout(height=600)
                st.plotly_chart(fig_map, use_container_width=True)
                
                # Market comparison
                st.subheader("Market Price Comparison")
                
                # Aggregate prices by market
                market_avg = st.session_state.filtered_data.groupby('market')['price'].agg(['mean', 'min', 'max', 'count']).round(2)
                market_avg = market_avg.sort_values('mean', ascending=False)
                
                fig_bar = px.bar(
                    market_avg.reset_index(),
                    x='market',
                    y='mean',
                    error_y=market_avg['max'] - market_avg['mean'],
                    error_y_minus=market_avg['mean'] - market_avg['min'],
                    title='Average Prices by Market',
                    labels={'mean': 'Average Price (ETB)', 'market': 'Market'},
                    color='mean',
                    color_continuous_scale='Viridis'
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Show market statistics
                st.subheader("Market Statistics")
                st.dataframe(market_avg, use_container_width=True)
            else:
                st.warning("Please apply filters to view market analysis")

if __name__ == "__main__":
    main()
