"""
Project: Marketing Budget Efficiency & Lead Conversion Audit
This project answers: Where should a business invest more, review, or stop spending in marketing
Author: Saad
Tools: Python, Pandas, NumPy, Matplotlib
"""
# ==========================================
#  Import Libraries
# ==========================================
import pandas as pd 
import numpy as np 
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

# ==========================================
#  Load Dataset
# ==========================================
# Loading the primary marketing performance dataset
df = pd.read_csv('/home/bear/projects/work/data/2nd project/Marketing.csv')

# ==========================================
#  Data Cleaning
# ==========================================

# Removing whitespace and standardizing headers for consistent data referencing
df.columns = df.columns.str.strip().str.lower()

# Verification of current column structure
print(df.columns.tolist())

# Auditing data types to ensure numerical values are ready for calculation
print('dtypes before converting to category: ',df.dtypes)

# Converting dates to datetime objects for accurate chronological analysis
df['c_date'] = pd.to_datetime(df['c_date'])

# Optimizing memory usage and processing speed by switching object types to categories
cols = df.select_dtypes(include=['object']).columns
df[cols] = df[cols].astype('category') 
print('dtypes after converting to category: ',df.dtypes)

# Data Integrity Check: Ensuring no critical info is missing before analysis
print(df.isnull().sum()) 

# Utility function to export the refined data for external reporting
def save_file(df):
    df.to_csv('cleaned_marketing.csv',index=False, float_format='%.2f')
    print('File Updated')

# ==========================================
#  Aggregated Analysis
# ==========================================
# Consolidating raw daily data into total performance metrics per campaign
campaign_group = df.groupby('campaign_name').agg({
    'impressions': 'sum',
    'mark_spent': 'sum',
    'clicks': 'sum',
    'leads': 'sum',
    'orders': 'sum',
    'revenue': 'sum'
}).reset_index()

# ==========================================
#  Feature Engineering (KPI Calculations)
# ==========================================

# CTR: Measures how effectively the ad creative captures audience interest
campaign_group['ctr'] = campaign_group['clicks'] / campaign_group['impressions']

# Lead Rate: Evaluates landing page effectiveness in converting traffic
campaign_group['lead_rate'] = campaign_group['leads'] / campaign_group['clicks']

# ROI: The ultimate profitability metric (Revenue generated per dollar spent)
campaign_group['roi'] = (campaign_group['revenue'] - campaign_group['mark_spent']) / campaign_group['mark_spent']

# Handling edge cases where zero clicks might cause calculation errors
campaign_group = campaign_group.fillna(0)

# Ranking campaigns by profitability to identify top performers
campaign_group = campaign_group.sort_values(by='roi', ascending=False)

# CPO: Shows the exact marketing cost required to secure a single customer order
campaign_group['cpo'] = campaign_group['mark_spent'] / campaign_group['orders']

# Correcting calculations for campaigns that spent budget but haven't yielded orders yet
campaign_group['cpo'] = campaign_group['cpo'].replace([np.inf, -np.inf], 0).fillna(0)
campaign_group['cpo'] = campaign_group['cpo'].round(2)

# Establishing performance benchmarks to compare individual campaigns against the average
average_spend = campaign_group['mark_spent'].mean()
average_roi = campaign_group['roi'].mean()
print(f"\nAverage Campaign Spend Benchmark: ${average_spend:.2f}")
print(f"Average ROI Benchmark: {average_roi:.2f}")

# --- Segmentation: Volume vs. Efficiency ---

print("\n--- Insight 1: Scalable 'Ideal' Campaigns (High Volume AND High ROI) ---")
# These are established winners where increasing the budget is low-risk
scalable_ideals = campaign_group[
    (campaign_group['mark_spent'] >= average_spend) & 
    (campaign_group['roi'] >= average_roi)
]
print('\n',scalable_ideals[['campaign_name', 'mark_spent', 'roi']])

print("\n--- Insight 2: 'Hidden Gems' (Low Volume BUT High ROI) ---")
# Highly efficient campaigns that could potentially explode with more investment
hidden_gems = campaign_group[
    (campaign_group['mark_spent'] < average_spend) & 
    (campaign_group['roi'] >= average_roi)
]
print('\n',hidden_gems[['campaign_name', 'mark_spent', 'roi']])

# High-level overview of all calculated performance lenses
print('\n',campaign_group[['campaign_name', 'mark_spent', 'ctr', 'lead_rate', 'cpo', 'roi']])

# Aggregating by marketing category to see which broad channels work best
campaign_group2 = df.groupby('category').agg({
    'impressions': 'sum',
    'mark_spent': 'sum',
    'clicks': 'sum',
    'leads': 'sum',
    'orders': 'sum',
    'revenue': 'sum'
}).reset_index()

# Performance calculations for category-level audit
campaign_group2['ctr'] = campaign_group2['clicks'] / campaign_group2['impressions']
campaign_group2['lead_rate'] = campaign_group2['leads'] / campaign_group2['clicks']
campaign_group2['roi'] = (campaign_group2['revenue'] - campaign_group2['mark_spent']) / campaign_group2['mark_spent']

# Order Share: Measures what percentage of total business sales each channel provides
total_orders = campaign_group2['orders'].sum()
campaign_group2['order_share_pct'] = (campaign_group2['orders'] / total_orders) * 100

# CPC: Monitoring the acquisition cost of traffic at the top of the funnel
campaign_group2['cpc'] = campaign_group2['mark_spent'] / campaign_group2['clicks']

print("\n--- Category Performance Audit: Volume vs. Efficiency ---")
print(campaign_group2[['category', 'mark_spent', 'order_share_pct', 'cpc', 'roi']].round(2))

# Final export of the dataset after all enhancements are added
save_file(df)

# ==========================================
#  Data Visualization
# ==========================================

# Chart 1: Visualizing Profitability across all individual campaigns
campaign_group_sorted = campaign_group.sort_values(by='roi', ascending=False)
campaign_group_sorted.plot(
    x='campaign_name',
    y='roi',
    kind='bar',
    figsize=(12,6)
)
plt.title("ROI Ranking by Marketing Campaign")
plt.ylabel("Return on Investment")
plt.xlabel("Campaign")
plt.xticks(rotation=45)
plt.tight_layout() 
plt.savefig('Visualizing_Profitability.png', dpi=300, bbox_inches='tight') 
plt.show()

# Chart 2: Direct comparison of Budget Invested vs. Total Revenue Generated
campaign_group_sorted_2 = campaign_group.sort_values(by='revenue', ascending=False)
campaign_group_sorted_2.plot(
    x='campaign_name',             
    y=['mark_spent', 'revenue'],    
    kind='bar',
    figsize=(12,6),
    color=["#a30f0f","#77c032"]     
)
plt.title("Marketing Campaign Spend vs. Revenue")
plt.ylabel("Amount ($)")
plt.xlabel("Campaign Name")
plt.xticks(rotation=45)
plt.legend(["Amount Spent", "Revenue Made"]) 
plt.tight_layout()
plt.savefig('Budget_invested_Vs_Total_Revenue.png', dpi=300, bbox_inches='tight') 
plt.show()

# Chart 3: Identifying the most profitable broad marketing channels
campaign_group_sorted_3 = campaign_group2.sort_values(by='roi', ascending=False)
campaign_group_sorted_3.plot(
    x='category',
    y='roi',
    kind='bar',
    color= ['green'],
    figsize=(12,6)
)
plt.title("ROI by Marketing Category")
plt.ylabel("Return on Investment")
plt.xlabel("Campaign")
plt.xticks(rotation=45)
plt.tight_layout() 
plt.savefig('Most_Profitable_Categories.png', dpi=300, bbox_inches='tight') 
plt.show()

# Chart 4: Analyzing the acquisition efficiency per campaign
campaign_group_cpo = campaign_group.sort_values(by='cpo')
campaign_group_cpo.plot(
    x='campaign_name',
    y='cpo',
    kind='bar',
    figsize=(12,6),
    color=["#1f5ab9"]
)
plt.title("Cost Per Order by Campaign")
plt.ylabel("Cost per Order ($)")
plt.xlabel("Campaign Name")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Acquisition_Efficiency_Campaign.png', dpi=300, bbox_inches='tight') 
plt.show()

# Chart 5: Budget Decision Matrix — Plotting Spend against Profit to find scaling opportunities
plt.figure(figsize=(10,6))
scatter = plt.scatter(
    campaign_group['mark_spent'],
    campaign_group['roi'],
    c=campaign_group['roi'], 
    cmap='RdYlGn', 
    edgecolor='black',
    s=100 
)
plt.colorbar(scatter, label="ROI Strength")

# Annotating points with campaign names for easy identification on the map
for i, txt in enumerate(campaign_group['campaign_name']):
    plt.annotate(
    txt,
    (campaign_group['mark_spent'].iloc[i], campaign_group['roi'].iloc[i]),
    xytext=(5, 5),             
    textcoords='offset points'
    )
# Final chart formatting and rendering
plt.title("Campaign ROI vs Marketing Spend")
plt.xlabel("Marketing Spend ($)")
plt.ylabel("ROI")
plt.grid(True)
plt.tight_layout()
plt.savefig('Budget_Decision_Matrix.png', dpi=300, bbox_inches='tight') 
plt.show()