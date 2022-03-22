# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Start

# %%
import pandas as pd
import numpy as np
from datetime import datetime

# %%
purchase_data = pd.read_excel("./Analyst_dataset.xlsx", sheet_name='Purchase Exit Survey Data')
airings_data = pd.read_excel("./Analyst_dataset.xlsx", sheet_name='Airings')

# The first row of Lookup table says "Lookup table for survey response field to airings network ticker symbol."  Assuming the first row always says that, we can drop it.
lookup_data = pd.read_excel("./Analyst_dataset.xlsx", sheet_name='Lookup', skiprows=1)

# %% [markdown]
# # Preprocessing

# %%
# Lookup data is meant to facilitate a join between the purchases data and the spend/lift data, but lookup data has a row will all null values, which doesn't help the join in any way.  I'll drop that row.
lookup_data = lookup_data.dropna(how='all')

# The Exit Survey column and Exit Survey.1 column are identical, so we can drop the duplicated column.
lookup_data.drop(labels='Exit Survey.1', axis=1, inplace=True)

# %%
# In order to make sure the joins happen correctly, we need to ensure that the strings we're joining on actually match.

# The purchase data table is pretty messy, but if we assume the second column always contains the names of the networks, we can use .iloc to grab them and ensure they're lowercase 

lookup_data['Exit Survey'] = lookup_data['Exit Survey'].str.lower()
lookup_data['Airings'] = lookup_data['Airings'].str.upper()
airings_data['Network'] = airings_data['Network'].str.upper()
purchase_data.iloc[:, 1] = purchase_data.iloc[:, 1].str.lower()

# %% [markdown]
# # Transposing Purchase Exit Survey Data - Converting dates from columns to rows in Purchase Exit Survey Data
#
# The purchases table is extremely messy and would be easier to work with if the rows were dates and the columns were the networks.  If we assume that the first row will always have the year, the third row will always have month names, and the fourth row will always have the day numbers, I can programmatically concatenate all the necessary date information in the form "Year-Month-Day", then use them for the rows.

# %%
current_year = purchase_data.iloc[0,:].dropna()
current_year = int(current_year)
current_year

# %%
months = []
for month in purchase_data.iloc[2,2:].dropna():
    months.append(month)
months

# %%
# Grab the row of day numbers and cast as integers
day_nums = np.array(purchase_data.iloc[3,2:], dtype=int)

parsed_dates = []
current_month = months[0]
i = 0

# Walk through the list of day_nums.  

# If current_day_num > next_day_num, that indicates a change in month (ex: If current_day = Sept-30 and next_day = Oct-1, b/c 30 > 1).  When this happens, we concatenate the current_day, then increase i by 1 to set the current_month to the next month for furture concatenation.  

# If current_day_num < next_day_num, that indicates both days are in the same month (ex: If current_day = Sept-5 and next_day = Sept-6, b/c 5 < 6), so we concatenate like normal.

# The try block handles the exception when you get to the last day in day_nums.  Since there are no more days in the list, we get an error when we try to index into the list one day into the future.
for count, current_day_num in enumerate(day_nums, start=1):
    try:
        next_day_num = day_nums[count]
    except:
        pass
    if current_day_num > next_day_num:
        current_date = str(current_year) + '-' + current_month + '-' + str(current_day_num)
        current_date = datetime.strptime(current_date, '%Y-%B-%d').date()
        i += 1
        current_month = months[i]
        parsed_dates.append(current_date)
    else:
        current_date = str(current_year) + '-' + current_month + '-' + str(current_day_num)
        current_date = datetime.strptime(current_date, '%Y-%B-%d').date()
        parsed_dates.append(current_date)

# Now that all the dates have been parsed, we replace the unparsed dates with the parsed ones, then transpose the table.  We now have rows that correspond to dates and columns that correspond to networks
purchase_data.iloc[3,2:] = parsed_dates
purchase_data_transpose = purchase_data.iloc[3:,:].transpose()

# %% [markdown]
# ## Some Cleanup

# %%
# Set the column of dates as the index and rename the axis appropriately
#purchase_data_transpose.index = purchase_data_transpose.iloc[:, 0]
purchase_data_transpose.set_index(3, inplace=True)
purchase_data_transpose.rename_axis('date', inplace=True)

# Drop first row, which doesn't contain anything useful
purchase_data_transpose = purchase_data_transpose.iloc[1:]

# Replace column names with the row of network names and then drop that row
purchase_data_transpose.columns = purchase_data_transpose.iloc[0]
purchase_data_transpose = purchase_data_transpose.drop(labels='source')

# Rename column axis as upper-case "Source" to match original table
purchase_data_transpose.rename_axis('Source', axis='columns', inplace=True)

# Convert index of dates to datetime objects
purchase_data_transpose.index = pd.to_datetime(purchase_data_transpose.index)

# %%
# purchase_data_transpose.shape

# %%
purchase_data_transpose.head()

# %% [markdown]
# ## Done

# %% [markdown]
# # Metrics by Network

# %% [markdown]
# ## Purchases by Network

# %%
purchases_by_network = purchase_data_transpose.sum(axis=0)
purchases_by_network = purchases_by_network.to_frame()
purchases_by_network = purchases_by_network.rename(columns={0:'Purchases'})

# %%
# purchases_by_network.shape

# %%
# purchases_by_network.head()

# %% [markdown]
# ## Spend and Lift by Network

# %%
spend_and_lift_by_network = airings_data.groupby('Network')[['Spend', 'Lift']].agg('sum')

# %%
# spend_and_lift_by_network.shape

# %%
# spend_and_lift_by_network.head()

# %% [markdown]
# ## Joins

# %% [markdown]
# ### Joining Purchases by Network to Lookup Data

# %%
purchases_by_network_w_lookup = lookup_data.merge(right=purchases_by_network, left_on='Exit Survey', right_on='Source', how='left')
purchases_by_network_w_lookup.set_index('Exit Survey', inplace=True)

# %%
# purchases_by_network_w_lookup.shape

# %%
# purchases_by_network_w_lookup.head()

# %% [markdown] tags=[]
# ### Joining Purchases/Lookup by Network to Spend and Lift

# %%
purchases_spend_lift_by_network = purchases_by_network_w_lookup.merge(right=spend_and_lift_by_network, left_on='Airings', right_index=True, how='left')

# Since this column was only needed for the join, I'm going to drop it post join
purchases_spend_lift_by_network.drop('Airings', axis=1, inplace=True)

# %%
# purchases_spend_lift_by_network.shape

# %%
# purchases_spend_lift_by_network.head()

# %%
purchases_spend_lift_by_network.index = purchases_spend_lift_by_network.index.str.replace('_', ' ').str.title()

# %%
purchases_spend_lift_by_network

# %%
purchases_spend_lift_by_network.fillna(0, inplace=True)

# %%
cnbc_world = airings_data[airings_data['Network']=='CNBCWORLD']

# %%
cnbc_world[cnbc_world['Date/Time ET'].isnull()==True]

# %% [markdown]
# ## Computing Metrics by Network

# %%
purchases_spend_lift_by_network['Conversion Rate (Purchases/Lift)%'] = purchases_spend_lift_by_network['Purchases'] / purchases_spend_lift_by_network['Lift'] * 100

purchases_spend_lift_by_network['Cost Per Acquisition (Spend/Purchases)'] = purchases_spend_lift_by_network['Spend'] / purchases_spend_lift_by_network['Purchases'].fillna(0)

purchases_spend_lift_by_network['Cost Per Visitor (Spend/Lift)'] = purchases_spend_lift_by_network['Spend'] / purchases_spend_lift_by_network['Lift'].fillna(0)

purchases_spend_lift_by_network['Percent of Purchases'] = purchases_spend_lift_by_network['Purchases'] / sum(purchases_spend_lift_by_network['Purchases'].fillna(0)) * 100

purchases_spend_lift_by_network['Percent of Spend'] = purchases_spend_lift_by_network['Spend'] / sum(purchases_spend_lift_by_network['Spend'].fillna(0)) * 100

purchases_spend_lift_by_network['Percent Pur > Percent Spend'] = purchases_spend_lift_by_network['Percent of Purchases'] > purchases_spend_lift_by_network['Percent of Spend']

# %%
purchases_spend_lift_by_network

# %%
# purchases_spend_lift_by_network

# %%
# purchases_spend_lift_by_network.shape

# %% [markdown]
# ## Output results to CSV file

# %%
current_year_and_months = str(current_year) + '_' + '_'.join(str(month) for month in months)

purchases_spend_lift_by_network.to_csv(F"./cleaned_output/purchases_spend_lift_by_network_{current_year_and_months}.csv")

# %% [markdown]
# ## Done

# %% [markdown]
# # Metrics by Network and Month

# %% [markdown]
# ## Purchases by Network and Month

# %%
# GroupBy month to find purchases per month for each network
purchase_data_by_month = purchase_data_transpose.groupby(pd.Grouper(freq='M')).agg('sum')

# %%
purchase_data_by_month

# %%
# Transpose so that the name of network becomes the rows and purchases per month become columns
purchase_data_by_month = purchase_data_by_month.transpose()

# Stack so that the index of each row corresponds to a network for a given month and row values are the number of purchases for that network in that month
purchase_data_by_month = purchase_data_by_month.stack().to_frame()

# Rename purchases column
purchase_data_by_month.rename(columns={0:'Purchases'}, inplace=True)

# Reset_index to set up for merging tables.  It will be easier to merge by column that by index in future steps
purchase_data_by_month = purchase_data_by_month.reset_index()

# %%
# purchase_data_by_month.head()

# %%
# purchase_data_by_month.shape

# %% [markdown]
# ## Spend and Lift by Network and Month

# %%
spend_lift_by_network_month = airings_data.groupby(['Network', pd.Grouper(key='Date/Time ET', freq='M')])[['Spend', 'Lift']].agg('sum')

# %% [markdown]
# ## Joins

# %% [markdown]
# ### Joining Purchases by Network and Month to Lookup Data

# %%
purchases_by_month_with_lookup = lookup_data.merge(right=purchase_data_by_month, left_on='Exit Survey', right_on='Source', how='left').set_index(['Exit Survey', 'date'])

# %%

# %%
# print(purchases_by_month_with_lookup.head().to_string())

# %%
# purchases_by_month_with_lookup.shape

# %% [markdown]
# ### Joining Purchases/Lookup by Network and Month to Spend and Lift

# %%
purchases_spend_lift_by_network_and_month = purchases_by_month_with_lookup.reset_index().merge(right=spend_lift_by_network_month.reset_index(), left_on=['Airings', 'date'], right_on=['Network', 'Date/Time ET'], how='left')

# %%
purchases_spend_lift_by_network_and_month.head()

# %%
# purchases_spend_lift_by_network_and_month.shape

# %%
purchases_spend_lift_by_network_and_month['Exit Survey'] = purchases_spend_lift_by_network_and_month['Exit Survey'].str.replace('_', ' ').str.title()

# %%
purchases_spend_lift_by_network_and_month.rename(columns={"Exit Survey": "Exit Survey Source"}, inplace=True)

# %%
purchases_spend_lift_by_network_and_month = purchases_spend_lift_by_network_and_month.set_index(['Exit Survey Source', 'date']).drop(labels=['Airings', 'Network', 'Date/Time ET', 'Source'], axis=1)

# %%
# purchases_spend_lift_by_network_and_month.head()

# %%
# purchases_spend_lift_by_network_and_month.shape

# %%
# print(purchases_spend_lift_by_network_and_month.head().to_string())

# %%
purchases_spend_lift_by_network_and_month.fillna(0, inplace=True)

# %% [markdown]
# ## Computing Metrics by Network and Month

# %%
purchases_spend_lift_by_network_and_month['Conversion Rate (Purchases/Lift)%'] = purchases_spend_lift_by_network_and_month['Purchases'] / purchases_spend_lift_by_network_and_month['Lift'] * 100

purchases_spend_lift_by_network_and_month['Cost Per Acquisition (Spend/Purchases)'] = purchases_spend_lift_by_network_and_month['Spend'] / purchases_spend_lift_by_network_and_month['Purchases']

purchases_spend_lift_by_network_and_month['Cost Per Visitor (Spend/Lift)'] = purchases_spend_lift_by_network_and_month['Spend'] / purchases_spend_lift_by_network_and_month['Lift']

purchases_spend_lift_by_network_and_month['Percent of Purchases'] = purchases_spend_lift_by_network_and_month['Purchases'] / sum(purchases_spend_lift_by_network_and_month['Purchases'].fillna(0)) * 100

purchases_spend_lift_by_network_and_month['Percent of Spend'] = purchases_spend_lift_by_network_and_month['Spend'] / sum(purchases_spend_lift_by_network_and_month['Spend'].fillna(0)) * 100

purchases_spend_lift_by_network_and_month['Percent Pur > Percent Spend'] = purchases_spend_lift_by_network_and_month['Percent of Purchases'] > purchases_spend_lift_by_network_and_month['Percent of Spend']

# %%
# purchases_spend_lift_by_network_and_month.head()

# %%
# print(purchases_spend_lift_by_network_and_month.head().to_string())

# %% [markdown]
# ## Output results to CSV file

# %%
purchases_spend_lift_by_network_and_month.to_csv(F"./cleaned_output/purchases_spend_lift_by_network_and_month_{current_year_and_months}.csv")

# %% [markdown]
# ## Done

# %%
# airings_data.index = pd.to_datetime(airings_data['Date/Time ET'])

# airings_data[airings_data['Network']=='COM'].groupby(pd.Grouper(freq='M')).sum()

# %% [markdown]
# # Generating Reports

# %% tags=[]
report_for_client = purchases_spend_lift_by_network.drop(['Percent of Purchases', 'Percent of Spend', 'Percent Pur > Percent Spend'], axis=1)#.dropna(how='all').fillna(0)

report_for_client.query('Spend != 0', inplace=True)

# %%
report_for_client

# %%
report_for_client_by_month = purchases_spend_lift_by_network_and_month.drop(['Percent of Purchases', 'Percent of Spend', 'Percent Pur > Percent Spend'], axis=1)#.dropna(how='all').fillna(0)

#report_for_client_by_month.query('Spend != 0', inplace=True)

# %%
report_for_client[['Purchases', 'Lift']] = report_for_client[['Purchases', 'Lift']].astype(int)

report_for_client = report_for_client.round({"Purchases":0, "Spend":2, "Lift":0, "Conversion Rate (Purchases/Lift)%":1, "Cost Per Acquisition (Spend/Purchases)":2, "Cost Per Visitor (Spend/Lift)":2})


report_for_client.rename_axis('Exit Survey Source', axis=0, inplace=True)

report_for_client = report_for_client.sort_values('Exit Survey Source')

# %%
report_for_client_by_month = report_for_client_by_month.round({"Purchases":0, "Spend":2, "Lift":0, "Conversion Rate (Purchases/Lift)%":1, "Cost Per Acquisition (Spend/Purchases)":2, "Cost Per Visitor (Spend/Lift)":2})

report_for_client_by_month[['Purchases', 'Lift']] = report_for_client_by_month[['Purchases', 'Lift']].astype(int)

# %%
report_for_client_by_month = report_for_client_by_month.loc[report_for_client.index]

# %% [markdown]
# ## Exporting Results to PDF Files

# %%
import pdfkit

f = open('./reports_output/html/report_for_client.html','w')
a = report_for_client.to_html(col_space='100px')
f.write(a)
f.close()

pdfkit.from_file('./reports_output/html/report_for_client.html', './reports_output/pdfs/report_for_client.pdf')

# %%
f = open('./reports_output/html/report_for_client_by_month.html','w')
a = report_for_client_by_month.to_html(col_space='100px')
f.write(a)
f.close()

pdfkit.from_file('./reports_output/html/report_for_client_by_month.html', './reports_output/pdfs/report_for_client_by_month.pdf')

# %% [markdown]
# ## Reports viewed in Jupyter Notebook

# %% [markdown]
# ### Report by Network

# %%
report_for_client

# %% [markdown]
# ### Report by Network and Month

# %%
report_for_client_by_month

# %% [markdown]
# ## Sorted Reports

# %% [markdown]
# ### Sorted by purchases

# %%
report_purchases_sorted = report_for_client.sort_values('Purchases', ascending=False)

# %%
f = open('./reports_output/html/report_purchases_sorted.html','w')
a = report_purchases_sorted.to_html(col_space='100px')
f.write(a)
f.close()

pdfkit.from_file('./reports_output/html/report_purchases_sorted.html', './reports_output/pdfs/report_purchases_sorted.pdf')

# %%
report_top_5_purchases = report_purchases_sorted[0:5]

# %% [markdown]
# ## Investigating reports

# %%

# %%

# %%

# %%
report_for_client_by_month.index

# %% [markdown]
# # Presentation 

# %% [markdown]
# ## How much does it cost to acquire a customer through TV?

# %%

# %%

# %% [markdown]
# ## Cost Efficiency

# %%
# For a channel to be cost-efficient, I'm going to limit my scope to only channels where money was spent on advertising there.
report_for_client[report_for_client['Spend']>0]['Cost Per Visitor (Spend/Lift)'].sort_values()[0:5].to_frame()

# %%
# For a channel to be cost-efficient, I'm going to limit my scope to only channels where money was spent on advertising there.
report_for_client[report_for_client['Spend']>0]['Cost Per Visitor (Spend/Lift)'].sort_values(ascending=False)[0:5].to_frame()

# %%
# For a channel to be cost-efficient, I'm going to limit my scope to only channels where money was spent on advertising there.

## Note: Both CNBC World and The Weather Channel had ZERO purchases, so despite the sort order, they technically have the worst Cost Per Acquisition
report_for_client[report_for_client['Spend']>0]['Cost Per Acquisition (Spend/Purchases)'].sort_values()[2:7].to_frame()

# %%
report_for_client[report_for_client['Spend']>0]['Cost Per Acquisition (Spend/Purchases)'].sort_values(ascending=False)

# %%
# For a channel to be cost-efficient, I'm going to limit my scope to only channels where money was spent on advertising there.

report_for_client[report_for_client['Spend']>0]['Conversion Rate (Purchases/Lift)%'].sort_values(ascending=False)

# %% [markdown]
# ## Which Networks have highest purchases?

# %%
num_purchases_by_network = purchases_spend_lift_by_network.groupby("Exit Survey")['Purchases'].agg('sum')

tot_purchases = sum(num_purchases_by_network)

percent_purchases_by_network = num_purchases_by_network / tot_purchases * 100
percent_purchases_by_network = percent_purchases_by_network.sort_values(ascending=False)

percent_purchases_by_network

# %% [markdown]
# ## Which Networks have no spend but high purchases?

# %%
num_purchases_no_spend = report_for_client[report_for_client['Spend']==0].groupby("Exit Survey Source")['Purchases'].agg('sum')

tot_purchases_no_spend = sum(num_purchases_no_spend)

perc_purchases_no_spend = num_purchases_no_spend / tot_purchases_no_spend * 100
perc_purchases_no_spend = perc_purchases_no_spend.sort_values(ascending=False)

perc_purchases_no_spend
# (report_for_client[report_for_client['Spend']==0].groupby("Exit Survey Source")['Purchases'].agg('sum') / sum(report_for_client[report_for_client['Spend']==0].groupby("Exit Survey Source")['Purchases'].agg('sum')) * 100).sort_values(ascending=False)

# %%

# %%

# %% [markdown]
# ## Need to Redo using report_for_client... - Overall Metrics - Establishing a baseline

# %%
total_purchases = sum(purchases_spend_lift_by_network['Purchases'].fillna(0))
total_spend = sum(purchases_spend_lift_by_network['Spend'].fillna(0))
total_lift = sum(purchases_spend_lift_by_network['Lift'].fillna(0))

# %%
mean_purchases = purchases_spend_lift_by_network['Purchases'].mean()
mean_spend = purchases_spend_lift_by_network['Spend'].mean()
mean_lift = purchases_spend_lift_by_network['Lift'].mean()

print(F"Average number purchases: {round(mean_purchases)}")
print(F"Average spend: ${round(mean_spend,2)}")
print(F"Average lift: {round(mean_lift)}")

# %%
mean_conversion_rate = purchases_spend_lift_by_network['Conversion Rate (Purchases/Lift)%'].mean()
mean_cpa = purchases_spend_lift_by_network['Cost Per Acquisition (Spend/Purchases)'].mean()
mean_cost_per_visitor = purchases_spend_lift_by_network['Cost Per Visitor (Spend/Lift)'].mean()

print(F"Average Cost Per Visitor: ${round(mean_cost_per_visitor)}")
print(F"Average Conversion Rate: {round(mean_conversion_rate,1)}%")
print(F"Average Cost Per Acquisition: ${round(mean_cpa,2)}")

# %%
report_for_client['Spend'].mean()

# %% [markdown]
# ### Cost Per Visitor = $ \frac{\text{total spend}}{\text{total lift}}$

# %%
cost_per_visitor = total_spend / total_lift
cost_per_visitor = round(cost_per_visitor, 2)
print(F"The overall cost per visitor for the TV campaign is ${cost_per_visitor}")

# %% [markdown]
# ### Conversion Rate = $ \frac{\text{total # of purchases}}{\text{total lift}} $

# %%
conversion_rate =  total_purchases / total_lift
conversion_rate = round(conversion_rate * 100, 2)
print(F"The overall conversion rate for the TV campaign is {conversion_rate}%")

# %% [markdown]
# ### Cost Per Acquisition = $ \frac{\text{total spend}}{\text{total # of purchases}} $

# %%
cost_per_acquisition = total_spend / total_purchases
cost_per_acquisition = round(cost_per_acquisition, 2)
print(F"The overall cost per acquisition for the TV campaign is ${cost_per_acquisition}")

# %%

# %%

# %%

# %%

# %% [markdown]
# # Graphing

# %%
import matplotlib.pyplot as plt

# %% [markdown]
# ## Bar Charts

# %%
ax = report_purchases_sorted[0:10].plot(kind='barh', y='Purchases', title='Top 10 Networks by Purchase', legend=False)
ax.invert_yaxis()
ax.set_xlabel('Number of Purchases');

# %%
ax = report_purchases_sorted[-10:].plot(kind='barh', y='Purchases', title='Bottom 10 Networks by Purchase', legend=False)
ax.invert_yaxis()
ax.set_xlabel('Number of Purchases')
ax.set_xticks(np.arange(0,3,1));

# %%
ax = report_purchases_sorted[0:10].plot(kind='barh', y='Purchases', title='Top 10 Networks by Purchase', legend=False, color='blue')
ax.invert_yaxis()
ax.set_xlabel('Number of Purchases')
ax1 = ax.twiny()
report_purchases_sorted[0:10].plot(kind='barh', y='Spend', title='Top 10 Networks by Purchase', legend=False, ax=ax, color='r');

# %%
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Number of Purchases')


ax2 = ax.twiny()
ax2.set_xlabel('Amount Spent ($)')


bar1 = report_purchases_sorted['Purchases'][0:10].plot(kind='barh', color='green', position=1, width=0.4, ax=ax, label='Purchases')
bar2 = report_purchases_sorted['Spend'][0:10].plot(kind='barh', color='red', position=0, width=0.4, ax=ax2, label='Spend')

#ax2.invert_yaxis()
plt.ylim((-0.5, len(report_purchases_sorted[0:10])-0.5))
ax.invert_yaxis()

bars, labels = ax.get_legend_handles_labels()
bars2, labels2 = ax2.get_legend_handles_labels()
ax.legend(bars+bars2, labels+labels2, loc='lower right')

;

# %%
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
ax.set_xlabel('Number of Purchases')


ax2 = ax.twiny()
ax2.set_xlabel('Amount Spent ($)')


report_purchases_sorted['Purchases'][-10:].plot(kind='barh', color='blue', position=1, width=0.4, ax=ax)
report_purchases_sorted['Spend'][-10:].plot(kind='barh', color='red', position=0, width=0.4, ax=ax2)
ax.invert_yaxis()
#ax2.invert_yaxis()
plt.ylim((-0.5, len(report_purchases_sorted[0:10])-0.5));

# %% [markdown]
# ## Heatmaps

# %%
import seaborn as sns

# %% [markdown]
# ## Purchases

# %%
fig, ax = plt.subplots(1,1,figsize=(2,10))
sns.heatmap(report_for_client[['Purchases']].sort_values(by='Purchases', ascending=False), annot=True, cmap='Greens', vmin=0, fmt='g')

# %%
fig, ax = plt.subplots(1,1,figsize=(2,10))
sns.heatmap(report_for_client[['Spend']].sort_values(by='Spend', ascending=False), annot=True, cmap='Reds', vmin=0, fmt='g')

# %%
fig, ax = plt.subplots(1,1,figsize=(2,10))
sns.heatmap(report_for_client[['Lift']].sort_values(by='Lift', ascending=False), annot=True, cmap='Reds', vmin=0, fmt='g')

# %%
fig, ax = plt.subplots(1,1,figsize=(2,10))
sns.heatmap(report_for_client[['Conversion Rate (Purchases/Lift)%']].sort_values(by='Conversion Rate (Purchases/Lift)%', ascending=False), annot=True, cmap='Reds', vmin=0, fmt='g')

# %%
fig, ax = plt.subplots(1,1,figsize=(2,10))
sns.heatmap(report_for_client[['Cost Per Acquisition (Spend/Purchases)']].sort_values(by='Cost Per Acquisition (Spend/Purchases)', ascending=False), annot=True, cmap='Reds', vmin=0, fmt='g')

# %%
fig, ax = plt.subplots(1,1,figsize=(2,10))
sns.heatmap(report_for_client[['Cost Per Visitor (Spend/Lift)']].sort_values(by='Cost Per Visitor (Spend/Lift)', ascending=False), annot=True, cmap='Reds', vmin=0, fmt='g')

# %%

# %%
#fig, ax = plt.subplots(1,1,figsize=(10,10))
sns.heatmap(report_for_client[['Cost Per Acquisition (Spend/Purchases)', 'Cost Per Visitor (Spend/Lift)']], annot=True, cmap='Reds', vmin=0, fmt='g')


# %%

# %%

# %% [markdown] tags=[]
# ## Plotting Function - make_scatter()

# %%
def make_scatter(df, x_field, y_field, x_units='', y_units='', color_1='green', color_2='red'):
    
    from adjustText import adjust_text
    
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    
    # ax.tick_params(top=False,
    #                bottom=True,
    #                left=True,
    #                right=False,
    #                labelleft=False,
    #                labelbottom=False)
    
    df[x_field] = df[x_field].replace(np.inf, 0)
    
    df.plot(kind='scatter', x=x_field, y=y_field, ax=ax)
    
    x_field_mean = df[x_field].mean()
    y_field_mean = df[y_field].mean()
    if x_units == "$":
        x_field_mean = round(x_field_mean, 2)
    elif x_units == "%":
        x_field_mean = round(x_field_mean, 1)
    else:
        x_field_mean = round(x_field_mean)
        
    if y_units == "$":
        y_field_mean = round(y_field_mean, 2)
    elif y_units == "%":
        y_field_mean = round(y_field_mean, 1)
    else:
        y_field_mean = round(y_field_mean)

    
    low_x_high_y = df[(df[x_field] < x_field_mean) & (df[y_field] >= y_field_mean)]
    high_x_low_y = df[(df[x_field] >= x_field_mean) & (df[y_field] < y_field_mean)]

    together = []
    
    for i in range(len(low_x_high_y)):
        txt1 = low_x_high_y.index[i]
        x_coord1 = low_x_high_y[x_field][i]
        y_coord1 = low_x_high_y[y_field][i]
        #size1 = low_x_high_y['Purchases'][i]
        color1 = color_1
        together.append((txt1, x_coord1, y_coord1, color1))
        ax.scatter(x_coord1, y_coord1, color=color1)

    for i in range(len(high_x_low_y)):
        txt2 = high_x_low_y.index[i]
        x_coord2 = high_x_low_y[x_field][i]
        y_coord2 = high_x_low_y[y_field][i]
        color2 = color_2
        together.append((txt2, x_coord2, y_coord2, color2))
        ax.scatter(x_coord2, y_coord2, color=color2)
    together.sort()


    text = [x for (x,y,z,c) in together]
    x_coords = [y for (x,y,z,c) in together]
    y_coords = [z for (x,y,z,c) in together]
    colors = [c for (x,y,z,c) in together]

    texts = []
    for x, y, s, c in zip(x_coords, y_coords, text, colors):
        texts.append(plt.text(x, y, s, color=c))

    
    # I should lookup how to do this with regex to make things easier...
    x_annot_text = x_field
    y_annot_text = y_field
    if x_field == "Conversion Rate (Purchases/Lift)%":
        x_annot_text = "Conversion Rate"
    elif x_field == "Cost Per Acquisition (Spend/Purchases)":
        x_annot_text = "Cost Per Acquisition"
    elif x_field == "Cost Per Visitor (Spend/Lift)":
        x_annot_text = "Cost Per Visitor"
        
    if y_field == "Conversion Rate (Purchases/Lift)%":
        y_annot_text = "Conversion Rate"
    elif y_field == "Cost Per Acquisition (Spend/Purchases)":
        y_annot_text = "Cost Per Acquisition"
    elif y_field == "Cost Per Visitor (Spend/Lift)":
        y_annot_text = "Cost Per Visitor"
    
    
    
    plt.axvline(x=x_field_mean, linestyle=(0, (2, 8)), color='k')
    ax.annotate(F'Mean {x_annot_text}'#: {round(x_field_mean, 2)}{x_units}'
                ,
                xy=(x_field_mean, max(ax.get_ylim())), xycoords='data',
                xytext=(0, 2), textcoords='offset pixels',
                color='k', ha='center')
    
    plt.axhline(y=y_field_mean, linestyle=(0, (2, 8)), color='k')
    ax.annotate(F'Mean\n{y_annot_text}' #:\n{round(y_field_mean, 2)}{y_units}'
                ,
                xy=(max(ax.get_xlim()), y_field_mean), xycoords='data',
                xytext=(5, 0), textcoords='offset pixels',
                color='k', ha='left')
    
    ax.axes.set_xticks([0, x_field_mean])
    #ax.axes.xaxis.set_ticklabels([])
    ax.axes.set_yticks([0, y_field_mean])
    #ax.axes.yaxis.set_ticklabels([])
    adjust_text(texts, 
            force_text=(0.5,2),
            force_points=(5,5),
            force_objects=(2,2),
            only_move={'points':'y', 'texts':'y'},
            arrowprops=dict(arrowstyle="->", color='k', lw=0.5))

    plt.show();

# %% [markdown]
# ## Scatter Plots

# %% [markdown]
# ### Using make_scatter()

# %% [markdown]
# #### Purchases vs. Spend

# %%
make_scatter(df=report_for_client,
             x_field='Purchases',
             y_field='Spend',
             #size=False,
             x_units='',
             y_units='$',
             color_1='red',
             color_2='green')

# %% [markdown]
# #### Lift vs. Spend

# %%
make_scatter(report_for_client,
             x_field='Lift',
             y_field='Spend',
             #size=True,
             x_units='',
             y_units='$',
             color_1='red',
             color_2='green')

# %% [markdown]
# #### Lift vs. Purchases

# %%
make_scatter(report_for_client,
             x_field='Lift',
             y_field='Purchases',
             x_units='',
             y_units='')

# %% [markdown]
# #### Conversion Rate vs. Spend

# %%
make_scatter(report_for_client,
             x_field='Conversion Rate (Purchases/Lift)%',
             y_field='Spend',
             x_units='%',
             y_units='$',
             color_1='red',
             color_2='green')

# %% [markdown]
# #### Conversion Rate vs. Cost Per Acquisition

# %%
make_scatter(report_for_client,
             x_field='Conversion Rate (Purchases/Lift)%',
             y_field='Cost Per Acquisition (Spend/Purchases)',
             x_units='%',
             y_units='$',
             color_1='red',
             color_2='green')

# %% [markdown]
# #### Conversion Rate vs. Cost Per Visitor

# %%
make_scatter(df=report_for_client,
             x_field="Conversion Rate (Purchases/Lift)%",
             y_field="Cost Per Visitor (Spend/Lift)",
             x_units="%",
             y_units="$",
             color_1='red',
             color_2='green')

# %%

# %%

# %%

# %% [markdown]
# ### Manual Plots

# %%
#Q1: High Purchases, High Spend, check cost per acquisition?
#Q2: Low Purchases, High Spend, Bad!
low_purchase_high_spend = report_for_client.query(F'(Purchases < {mean_purchases}) & (Spend >= {mean_spend})').sort_values('Cost Per Acquisition (Spend/Purchases)', ascending=False)
#Q3: Low Purchases, Low Spend, check cost per acquisition?
#Q4: High Purchases, Low Spend, great!
high_purchase_low_spend = report_for_client.query(F'(Purchases >= {mean_purchases}) & (Spend < {mean_spend})').sort_values('Cost Per Acquisition (Spend/Purchases)', ascending=False)

fig, ax = plt.subplots(1,1,figsize=(10,5))
report_for_client[['Purchases', 'Spend']].plot(kind='scatter', x='Purchases', y='Spend', ax=ax)

plt.tick_params(left=True,labelleft=False,
                   labelbottom=False)

for i in range(len(low_purchase_high_spend)):
    txt = low_purchase_high_spend.index[i]
    x_coord = low_purchase_high_spend['Purchases'][i]
    y_coord = low_purchase_high_spend['Spend'][i]
    ax.annotate(txt, xy=(x_coord, y_coord), xytext=(x_coord, y_coord+1_000), color='red', ha='center')
    ax.scatter(x_coord, y_coord, color='red')
    
for i in range(len(high_purchase_low_spend)):
    txt = high_purchase_low_spend.index[i]
    x_coord = high_purchase_low_spend['Purchases'][i]
    y_coord = high_purchase_low_spend['Spend'][i]
    ax.scatter(x_coord, y_coord, color='green')
    if i == 3:
        ax.annotate(txt, xy=(x_coord, y_coord), xytext=(x_coord-0.5, y_coord+1_000), color='green', ha='right')
    elif i == 2:
        ax.annotate(txt, xy=(x_coord, y_coord), xytext=(x_coord+0.5, y_coord+1_000), color='green', ha='left')
    else:
        ax.annotate(txt, xy=(x_coord, y_coord), xytext=(x_coord+0.5, y_coord+1_000), color='green', ha='center')

#report_top_5_purchases[['Purchases', 'Spend']].plot(kind='scatter', x='Purchases', y='Spend', color='red', ax=ax)
plt.axhline(y=mean_spend, linestyle='--', color='k')
plt.text(x=45, y=mean_spend+1_000, s=F'Mean Spend: ${round(mean_spend,2)}', color='k')
plt.axvline(x=mean_purchases, linestyle='--', color='k')
plt.text(x=mean_purchases+0.5, y=48_000, s=F'Mean Purchases: {round(mean_purchases)}', color='k')

# plt.text(x=2, y=45_000, s='Low Purchases,', ha='center', color='red')
# plt.text(x=2, y=43_000, s='High Spend', ha='center', color='red')

# plt.text(x=40, y=5_000, s='High Purchases,', ha='center', color='green')
# plt.text(x=40, y=3_000, s='Low Spend', ha='center', color='green')

plt.show();

# %% tags=[]
report_for_client.query(F'(Purchases >= {mean_purchases}) & (Spend >= {mean_spend})').sort_values('Cost Per Acquisition (Spend/Purchases)', ascending=False)

# %%
report_for_client.query(F'(Purchases < {mean_purchases}) & (Spend >= {mean_spend})').sort_values('Cost Per Acquisition (Spend/Purchases)', ascending=False)

# %% tags=[]
report_for_client.query(F'(Purchases < {mean_purchases}) & (Spend < {mean_spend})').sort_values('Cost Per Acquisition (Spend/Purchases)', ascending=False)

# %%
report_for_client.query(F'(Purchases >= {mean_purchases}) & (Spend < {mean_spend})').sort_values('Cost Per Acquisition (Spend/Purchases)', ascending=False)

# %%
#Q1: High Lift, High Spend, check cost per visitor?
#Q2: Low Lift, High Spend, Bad!
#Q3: Low Lift, Low Spend, check cost per visitor?
#Q4: High Lift, Low Spend great!

fig, ax = plt.subplots(1,1,figsize=(10,5))
report_for_client.plot(kind='scatter', x='Lift', y='Spend', ax=ax)
#report_top_5_purchases.plot(kind='scatter', x='Lift', y='Spend', color='red', ax=ax)

low_lift_high_spend = report_for_client.query(F'(Lift < {mean_lift}) & (Spend >= {mean_spend})').sort_values('Cost Per Visitor (Spend/Lift)', ascending=False)

for i in range(len(low_lift_high_spend)):
    txt = low_lift_high_spend.index[i]
    x_coord = low_lift_high_spend['Lift'][i]
    y_coord = low_lift_high_spend['Spend'][i]
    ax.annotate(txt, xy=(x_coord, y_coord), xytext=(x_coord, y_coord+1_000), color='red', ha='right')
    ax.scatter(x_coord, y_coord, color='red')

high_lift_low_spend = report_for_client.query(F'(Lift >= {mean_lift}) & (Spend < {mean_spend})').sort_values('Cost Per Visitor (Spend/Lift)', ascending=False)

for i in range(len(high_lift_low_spend)):
    txt = high_lift_low_spend.index[i]
    x_coord = high_lift_low_spend['Lift'][i]
    y_coord = high_lift_low_spend['Spend'][i]
    ax.annotate(txt, xy=(x_coord, y_coord), xytext=(x_coord+50, y_coord-2_000), color='green', ha='left')
    ax.scatter(x_coord, y_coord, color='green')

# plt.axhline(y=mean_spend)
# plt.axvline(x=mean_lift)

plt.axhline(y=mean_spend, linestyle='--', color='k')
plt.text(x=6_500, y=mean_spend+1_000, s=F'Mean Spend: ${round(mean_spend,2)}', color='k')

plt.axvline(x=mean_lift, linestyle='--', color='k')
plt.text(x=mean_lift+100, y=48_000, s=F'Mean Lift: {round(mean_lift)}', color='k')

plt.show();

# %% tags=[]
report_for_client.query(F'(Lift >= {mean_lift}) & (Spend >= {mean_spend})').sort_values('Cost Per Visitor (Spend/Lift)', ascending=False)

# %%
report_for_client.query(F'(Lift < {mean_lift}) & (Spend >= {mean_spend})').sort_values('Cost Per Visitor (Spend/Lift)', ascending=False)

# %% tags=[]
report_for_client.query(F'(Lift < {mean_lift}) & (Spend < {mean_spend})').sort_values('Cost Per Visitor (Spend/Lift)', ascending=False)

# %%
report_for_client.query(F'(Lift >= {mean_lift}) & (Spend < {mean_spend})').sort_values('Cost Per Visitor (Spend/Lift)', ascending=False)

# %%
#Q1: High Lift, High Purchases, check _?
#Q2: Low Lift, High Purchases, check spend, maybe spend more here!
#Q3: Low Lift, Low Purchases, check conversion rate, spend more on ones that have high conversion rate?  Maybe also check cost per visitor?
#Q4: High Lift, Low Purchases, not good?
fig, ax = plt.subplots(1,1,figsize=(10,5))
report_for_client.plot(kind='scatter', x='Lift', y='Purchases', ax=ax)

low_lift_high_purchases = report_for_client.query(F'(Lift < {mean_lift}) & (Purchases >= {mean_purchases})').sort_values('Cost Per Visitor (Spend/Lift)', ascending=False)

for i in range(len(low_lift_high_purchases)):
    txt = low_lift_high_purchases.index[i]
    x_coord = low_lift_high_purchases['Lift'][i]
    y_coord = low_lift_high_purchases['Purchases'][i]
    if i == 1: #cnbc
        ax.annotate(txt, xy=(x_coord, y_coord), xytext=(x_coord, y_coord+2), color='green', ha='left')
        ax.scatter(x_coord, y_coord, color='green')
    elif i == 2: #fox news
        ax.annotate(txt, xy=(x_coord, y_coord), xytext=(x_coord, y_coord+1), color='green', ha='center')
        ax.scatter(x_coord, y_coord, color='green')
    elif i == 3: #other
        ax.annotate(txt, xy=(x_coord, y_coord), xytext=(x_coord, y_coord-3), color='green', ha='center')
        ax.scatter(x_coord, y_coord, color='green')
    else: #cnbc and dateline
        ax.annotate(txt, xy=(x_coord, y_coord), xytext=(x_coord, y_coord+1), color='green', ha='center')
        ax.scatter(x_coord, y_coord, color='green')

high_lift_low_purchases = report_for_client.query(F'(Lift >= {mean_lift}) & (Purchases < {mean_purchases})').sort_values('Cost Per Visitor (Spend/Lift)', ascending=False)

for i in range(len(high_lift_low_purchases)):
    txt = high_lift_low_purchases.index[i]
    x_coord = high_lift_low_purchases['Lift'][i]
    y_coord = high_lift_low_purchases['Purchases'][i]
    
    if i == 2: #dishnetwork
        ax.annotate(txt, xy=(x_coord, y_coord), xytext=(x_coord+140, y_coord-5), color='red', ha='center')
        ax.scatter(x_coord, y_coord, color='red')
    elif i == 0: #zeetv
        ax.annotate(txt, xy=(x_coord, y_coord), xytext=(x_coord, y_coord+1), color='red', ha='center')
        ax.scatter(x_coord, y_coord, color='red')
    else: #starplus
        ax.annotate(txt, xy=(x_coord, y_coord), xytext=(x_coord+100, y_coord-2), color='red', ha='left')
        ax.scatter(x_coord, y_coord, color='red')

# report_top_5_purchases.plot(kind='scatter', x='Lift', y='Purchases', color='red', ax=ax)
# plt.axhline(y=mean_purchases)
# plt.axvline(x=mean_lift)

plt.axhline(y=mean_purchases, linestyle='--', color='k')
plt.text(x=6_500, y=mean_purchases+1, s=F'Mean Purchases: {round(mean_purchases)}', color='k')

plt.axvline(x=mean_lift, linestyle='--', color='k')
plt.text(x=mean_lift+100, y=55, s=F'Mean Lift: {round(mean_lift)}', color='k')

plt.show();

# %%
report_for_client.query(F'(Lift < {mean_lift}) & (Purchases >= {mean_purchases})').sort_values('Cost Per Visitor (Spend/Lift)', ascending=False)

# %%
report_for_client.query(F'(Lift >= {mean_lift}) & (Purchases < {mean_purchases})').sort_values('Cost Per Visitor (Spend/Lift)', ascending=False)

# %%
#Q1: High Conversion Rate, High Spend, check _?
#Q2: Low Conversion Rate, High Spend, bad!
#Q3: Low Conversion Rate, Low Spend, check conversion rate, check cost per acquisition and cost per visitor?
#Q4: high Conversion Rate, Low Spend great!

fig, ax = plt.subplots(1,1,figsize=(10,5))
report_for_client.plot(kind='scatter', x='Conversion Rate (Purchases/Lift)%', y='Spend', ax=ax)

low_conversion_rate_high_spend = report_for_client[(report_for_client["Conversion Rate (Purchases/Lift)%"] < mean_conversion_rate) & (report_for_client["Spend"] >= mean_spend)]

for i in range(len(low_conversion_rate_high_spend)):
    txt = low_conversion_rate_high_spend.index[i]
    x_coord = low_conversion_rate_high_spend['Conversion Rate (Purchases/Lift)%'][i]
    y_coord = low_conversion_rate_high_spend['Spend'][i]
    ax.annotate(txt, xy=(x_coord, y_coord), xytext=(x_coord, y_coord+1_000), color='red', ha='center')
    ax.scatter(x_coord, y_coord, color='red')

    
high_conversion_rate_low_spend = report_for_client[(report_for_client["Conversion Rate (Purchases/Lift)%"] >= mean_conversion_rate) & (report_for_client["Spend"] < mean_spend)]
    
# report_top_5_purchases.plot(kind='scatter', x='Conversion Rate (Purchases/Lift)%', y='Spend', color='red', ax=ax)
# plt.axhline(y=mean_spend)
# plt.axvline(x=mean_conversion_rate)

plt.axhline(y=mean_spend, linestyle='--', color='k')
plt.text(x=3, y=mean_spend+1_000, s=F'Mean Spend: ${round(mean_spend, 2)}', color='k')

plt.axvline(x=mean_conversion_rate, linestyle='--', color='k')
plt.text(x=mean_conversion_rate+0.05, y=45_000, s=F'Mean Conversion Rate: {round(mean_conversion_rate, 1)}%', color='k')

plt.show();

# %% [markdown]
# #### Where you started using adjustText

# %%
from adjustText import adjust_text

fig, ax = plt.subplots(1,1,figsize=(10,5))
report_for_client.plot(kind='scatter', x='Conversion Rate (Purchases/Lift)%', y='Spend', ax=ax)

together1 = []
# text_list = []
# x_coords = []
# y_coords = []
for i in range(len(low_conversion_rate_high_spend)):
    txt1 = low_conversion_rate_high_spend.index[i]
    x_coord1 = low_conversion_rate_high_spend['Conversion Rate (Purchases/Lift)%'][i]
    y_coord1 = low_conversion_rate_high_spend['Spend'][i]
    together1.append((txt1, x_coord1, y_coord1))
    ax.scatter(x_coord1, y_coord1, color='red')
together1.sort()

text1 = [x for (x,y,z) in together1]
x_coords1 = [y for (x,y,z) in together1]
y_coords1 = [z for (x,y,z) in together1]

texts1 = []
for x, y, s in zip(x_coords1, y_coords1, text1):
    texts1.append(plt.text(x, y, s, color='red'))

adjust_text(texts1, only_move={'points':'y', 'texts1':'y'})


together2 = []
# text_list = []
# x_coords = []
# y_coords = []
for i in range(len(high_conversion_rate_low_spend)):
    txt2 = high_conversion_rate_low_spend.index[i]
    x_coord2 = high_conversion_rate_low_spend['Conversion Rate (Purchases/Lift)%'][i]
    y_coord2 = high_conversion_rate_low_spend['Spend'][i]
    together2.append((txt2, x_coord2, y_coord2))
    ax.scatter(x_coord2, y_coord2, color='green')
together2.sort()

text2 = [x for (x,y,z) in together2]
x_coords2 = [y for (x,y,z) in together2]
y_coords2 = [z for (x,y,z) in together2]

texts2 = []
for x, y, s in zip(x_coords2, y_coords2, text2):
    texts2.append(plt.text(x, y, s, color='green'))

adjust_text(texts2, only_move={'points':'y', 'texts2':'y'})

plt.axhline(y=mean_spend, linestyle=(0, (2, 8)), color='k')
plt.text(x=2.9, y=mean_spend+1_000, s=F'Mean Spend: ${round(mean_spend, 2)}', color='k')

plt.axvline(x=mean_conversion_rate, linestyle=(0, (2, 8)), color='k')
plt.text(x=mean_conversion_rate+0.05, y=45_000, s=F'Mean Conversion Rate: {round(mean_conversion_rate, 1)}%', color='k')

plt.show()

# %%
from adjustText import adjust_text

fig, ax = plt.subplots(1,1,figsize=(10,5))
report_for_client.plot(kind='scatter', x='Conversion Rate (Purchases/Lift)%', y='Spend', ax=ax)

together = []

for i in range(len(low_conversion_rate_high_spend)):
    txt1 = low_conversion_rate_high_spend.index[i]
    x_coord1 = low_conversion_rate_high_spend['Conversion Rate (Purchases/Lift)%'][i]
    y_coord1 = low_conversion_rate_high_spend['Spend'][i]
    color1 = 'red'
    together.append((txt1, x_coord1, y_coord1, color1))
    ax.scatter(x_coord1, y_coord1, color=color1)

for i in range(len(high_conversion_rate_low_spend)):
    txt2 = high_conversion_rate_low_spend.index[i]
    x_coord2 = high_conversion_rate_low_spend['Conversion Rate (Purchases/Lift)%'][i]
    y_coord2 = high_conversion_rate_low_spend['Spend'][i]
    color2 = 'green'
    together.append((txt2, x_coord2, y_coord2, color2))
    ax.scatter(x_coord2, y_coord2, color=color2)
together.sort()

text = [x for (x,y,z,c) in together]
x_coords = [y for (x,y,z,c) in together]
y_coords = [z for (x,y,z,c) in together]
colors = [c for (x,y,z,c) in together]

texts = []
for x, y, s, c in zip(x_coords, y_coords, text, colors):
    texts.append(plt.text(x, y, s, color=c))

adjust_text(texts, only_move={'points':'y', 'texts':'y'})


plt.axhline(y=mean_spend, linestyle=(0, (2, 8)), color='k')
plt.text(x=2.9, y=mean_spend+1_000, s=F'Mean Spend: ${round(mean_spend, 2)}', color='k')

plt.axvline(x=mean_conversion_rate, linestyle=(0, (2, 8)), color='k')
plt.text(x=mean_conversion_rate+0.05, y=45_000, s=F'Mean Conversion Rate: {round(mean_conversion_rate, 1)}%', color='k')

plt.show()

# %%

# %%
report_for_client[(report_for_client["Conversion Rate (Purchases/Lift)%"] < mean_conversion_rate) & (report_for_client["Spend"] >= mean_spend)]

# %%
report_for_client[(report_for_client["Conversion Rate (Purchases/Lift)%"] >= mean_conversion_rate) & (report_for_client["Spend"] < mean_spend)]

# %%
# report_for_client.query("(Conversion Rate (Purchases/Lift)% < {0}) & (Spend >= {1})".format(mean_conversion_rate, mean_spend)).sort_values('Cost Per Visitor (Spend/Lift)', ascending=False)

# %%
# # This chart just seems bad

# #Q1: High Cost Per Acquisition, High Spend, bad!
# #Q2: Low Cost Per Acquisition, High Spend, must mean lots of purchases?  Doesn't make a lot of sense
# #Q3: Low Cost Per Acquisition, Low Spend, check conversion rate, check cost per acquisition and cost per visitor?
# #Q4: Low Spend, high Cost Per Acquisition, doesn't make a lot of sense..?
# fig, ax = plt.subplots(1,1,figsize=(10,5))
# report_for_client.plot(kind='scatter', x='Cost Per Acquisition', y='Spend', ax=ax)
# report_top_5_purchases.plot(kind='scatter', x='Cost Per Acquisition', y='Spend', color='red', ax=ax)
# plt.axhline(y=mean_spend)
# plt.axvline(x=mean_cpa)
# ;

# %%
# # This chart just seems bad

# #Q1: High Cost Per Visitor, High Spend, none present, doesn't matter
# #Q2: Low Cost Per Visitor, High Spend, Seems weird? Doesn't make sense
# #Q3: Low Cost Per Visitor, Low Spend, Seems weird, doesn't make sense
# #Q4: High Cost Per Visitor, Low Spend, seems weird, doesn't make sense

# fig, ax = plt.subplots(1,1,figsize=(10,5))
# report_for_client.plot(kind='scatter', x='Cost Per Visitor', y='Spend', ax=ax)
# report_top_5_purchases.plot(kind='scatter', x='Cost Per Visitor', y='Spend', color='red', ax=ax)
# plt.axhline(y=mean_spend)
# plt.axvline(x=mean_cost_per_visitor)
# ;

# %%
#Q1: High Conversion Rate, High Cost Per Acquisition, check _?
#Q2: Low Conversion Rate, High Cost Per Acquisition, bad!
#Q3: Low Conversion Rate, Low Cost Per Acquisition, unsure?
#Q4: High Conversion Rate, Low Cost Per Acquisition, great!

fig, ax = plt.subplots(1,1,figsize=(10,5))
report_for_client.plot(kind='scatter', x='Conversion Rate (Purchases/Lift)%', y='Cost Per Acquisition (Spend/Purchases)', ax=ax)


report_top_5_purchases.plot(kind='scatter', x='Conversion Rate (Purchases/Lift)%', y='Cost Per Acquisition (Spend/Purchases)', color='red', ax=ax)
plt.axhline(y=mean_cpa)
plt.axvline(x=mean_conversion_rate)

plt.show();

# %%
make_scatter(df=report_for_client,
             x_field="Conversion Rate (Purchases/Lift)%",
             y_field="Cost Per Acquisition (Spend/Purchases)",
             x_units="%",
             y_units="$")

# %%
fig, ax = plt.subplots(1,1,figsize=(10,5))
report_for_client.plot(kind='scatter', x='Conversion Rate (Purchases/Lift)%', y='Cost Per Acquisition (Spend/Purchases)', ax=ax)

low_conversion_rate_high_cost_per_acquisition = report_for_client[(report_for_client["Conversion Rate (Purchases/Lift)%"] < mean_conversion_rate) & (report_for_client["Cost Per Acquisition (Spend/Purchases)"] >= mean_cpa)]

together = []

for i in range(len(low_conversion_rate_high_cost_per_acquisition)):
    txt1 = low_conversion_rate_high_cost_per_acquisition.index[i]
    x_coord1 = low_conversion_rate_high_cost_per_acquisition['Conversion Rate (Purchases/Lift)%'][i]
    y_coord1 = low_conversion_rate_high_cost_per_acquisition['Cost Per Acquisition (Spend/Purchases)'][i]
    color1 = 'red'
    together.append((txt1, x_coord1, y_coord1, color1))
    ax.scatter(x_coord1, y_coord1, color=color1)


high_conversion_rate_low_cost_per_acquisition = report_for_client[(report_for_client["Conversion Rate (Purchases/Lift)%"] >= mean_conversion_rate) & (report_for_client["Cost Per Acquisition (Spend/Purchases)"] < mean_cpa)]


for i in range(len(high_conversion_rate_low_cost_per_acquisition)):
    txt2 = high_conversion_rate_low_cost_per_acquisition.index[i]
    x_coord2 = high_conversion_rate_low_cost_per_acquisition['Conversion Rate (Purchases/Lift)%'][i]
    y_coord2 = high_conversion_rate_low_cost_per_acquisition['Cost Per Acquisition (Spend/Purchases)'][i]
    color2 = 'green'
    together.append((txt2, x_coord2, y_coord2, color2))
    ax.scatter(x_coord2, y_coord2, color=color2)
together.sort()

    
text = [x for (x,y,z,c) in together]
x_coords = [y for (x,y,z,c) in together]
y_coords = [z for (x,y,z,c) in together]
colors = [c for (x,y,z,c) in together]

texts = []
for x, y, s, c in zip(x_coords, y_coords, text, colors):
    texts.append(plt.text(x, y, s, color=c))

adjust_text(texts, only_move={'points':'y', 'texts':'y'})


plt.axhline(y=mean_cpa, linestyle=(0, (2, 8)), color='k')
plt.text(x=2.5, y=mean_cpa-300, s=F'Mean Cost Per Acquisition: ${round(mean_cpa, 2)}', color='k')

plt.axvline(x=mean_conversion_rate, linestyle=(0, (2, 8)), color='k')
plt.text(x=mean_conversion_rate+0.05, y=5_000, s=F'Mean Conversion Rate: {round(mean_conversion_rate, 1)}%', color='k')

plt.show();

# %%
report_for_client[(report_for_client["Conversion Rate (Purchases/Lift)%"] < mean_conversion_rate) & (report_for_client["Cost Per Acquisition (Spend/Purchases)"] >= mean_cpa)]

# %%
report_for_client[(report_for_client["Conversion Rate (Purchases/Lift)%"] >= mean_conversion_rate) & (report_for_client["Cost Per Acquisition (Spend/Purchases)"] < mean_cpa)]

# %%
#Q1: High Conversion Rate, High Cost Per Visitor, check _?
#Q2: Low Conversion Rate, High Cost Per Visitor, bad!
#Q3: Low Conversion Rate, Low Cost Per Visitor, unsure?
#Q4: High Conversion Rate, Low Cost Per Visitor, great!

fig, ax = plt.subplots(1,1,figsize=(10,5))
report_for_client.plot(kind='scatter', x='Conversion Rate (Purchases/Lift)%', y='Cost Per Visitor (Spend/Lift)', ax=ax)
report_top_5_purchases.plot(kind='scatter', x='Conversion Rate (Purchases/Lift)%', y='Cost Per Visitor (Spend/Lift)', color='red', ax=ax)
plt.axhline(y=mean_cost_per_visitor)
plt.axvline(x=mean_conversion_rate)
;

# %%
fig, ax = plt.subplots(1,1,figsize=(10,5))
report_for_client.plot(kind='scatter', x='Conversion Rate (Purchases/Lift)%', y='Cost Per Visitor (Spend/Lift)', ax=ax)

low_conversion_rate_high_cost_per_visitor = report_for_client[(report_for_client["Conversion Rate (Purchases/Lift)%"] < mean_conversion_rate) & (report_for_client["Cost Per Visitor (Spend/Lift)"] >= mean_cost_per_visitor)]

together1 = []
# text_list = []
# x_coords = []
# y_coords = []
for i in range(len(low_conversion_rate_high_cost_per_visitor)):
    txt1 = low_conversion_rate_high_cost_per_visitor.index[i]
    x_coord1 = low_conversion_rate_high_cost_per_visitor['Conversion Rate (Purchases/Lift)%'][i]
    y_coord1 = low_conversion_rate_high_cost_per_visitor['Cost Per Visitor (Spend/Lift)'][i]
    together1.append((txt1, x_coord1, y_coord1))
    ax.scatter(x_coord1, y_coord1, color='red')
together1.sort()

text1 = [x for (x,y,z) in together1]
x_coords1 = [y for (x,y,z) in together1]
y_coords1 = [z for (x,y,z) in together1]

texts1 = []
for x, y, s in zip(x_coords1, y_coords1, text1):
    texts1.append(plt.text(x, y, s, color='red'))

adjust_text(texts1, only_move={'points':'y', 'texts1':'y'})

high_conversion_rate_low_cost_per_visitor = report_for_client[(report_for_client["Conversion Rate (Purchases/Lift)%"] >= mean_conversion_rate) & (report_for_client["Cost Per Visitor (Spend/Lift)"] < mean_cost_per_visitor)]

together2 = []
# text_list = []
# x_coords = []
# y_coords = []
for i in range(len(high_conversion_rate_low_cost_per_visitor)):
    txt2 = high_conversion_rate_low_cost_per_visitor.index[i]
    x_coord2 = high_conversion_rate_low_cost_per_visitor['Conversion Rate (Purchases/Lift)%'][i]
    y_coord2 = high_conversion_rate_low_cost_per_visitor['Cost Per Visitor (Spend/Lift)'][i]
    together2.append((txt2, x_coord2, y_coord2))
    ax.scatter(x_coord2, y_coord2, color='green')
together2.sort()

text2 = [x for (x,y,z) in together2]
x_coords2 = [y for (x,y,z) in together2]
y_coords2 = [z for (x,y,z) in together2]

texts2 = []
for x, y, s in zip(x_coords2, y_coords2, text2):
    texts2.append(plt.text(x, y, s, color='green'))

adjust_text(texts2, only_move={'points':'y', 'texts2':'y'})

plt.axhline(y=mean_cost_per_visitor, linestyle=(0, (2, 8)), color='k')
plt.text(x=2.75, y=mean_cost_per_visitor+2, s=F'Mean Cost Per Visitor: ${round(mean_cost_per_visitor, 2)}', color='k')

plt.axvline(x=mean_conversion_rate, linestyle=(0, (2, 8)), color='k')
plt.text(x=mean_conversion_rate+0.05, y=65, s=F'Mean Conversion Rate: {round(mean_conversion_rate, 1)}%', color='k')

plt.show();

# %%
report_for_client[(report_for_client["Conversion Rate (Purchases/Lift)%"] < mean_conversion_rate) & (report_for_client["Cost Per Visitor (Spend/Lift)"] >= mean_cost_per_visitor)]

# %%
report_for_client[(report_for_client["Conversion Rate (Purchases/Lift)%"] >= mean_conversion_rate) & (report_for_client["Cost Per Visitor (Spend/Lift)"] < mean_cost_per_visitor)]

# %%
# # This graph seems bad

# #Q1: High Cost Per Acquisition, High Cost Per Visitor, sounds bad?
# #Q2: Low Cost Per Acquisition, High Cost Per Visitor, sounds weird!
# #Q3: Low Cost Per Acquisition, Low Cost Per Visitor, unsure?
# #Q4: High Cost Per Acquisition, Low Cost Per Visitor, sounds weird!

# fig, ax = plt.subplots(1,1,figsize=(10,5))
# report_for_client.plot(kind='scatter', x='Cost Per Acquisition (Spend/Purchases)', y='Cost Per Visitor (Spend/Lift)', ax=ax)
# report_top_5_purchases.plot(kind='scatter', x='Cost Per Acquisition (Spend/Purchases)', y='Cost Per Visitor (Spend/Lift)', color='red', ax=ax)
# plt.axhline(y=mean_cost_per_visitor)
# plt.axvline(x=mean_cpa)
# ;

# %%
# fix, ax = plt.subplots(1,1, figsize=(10,5))
# for column in report_for_client.columns:
#     report_for_client.plot.box(x='Purchases', ax=ax, vert=False)


#fix, ax = plt.subplots(1,1, figsize=(10,5))
for column in report_for_client.columns:
    #print(column)
    fig,ax = plt.subplots(1,1)
    report_for_client[column].plot.box(vert=False, ax=ax)

# %% [markdown]
# # Done

# %%

# %%
