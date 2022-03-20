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
# purchase_data_transpose.head()

# %% [markdown]
# ## Done

# %% [markdown]
# # Metrics by Network

# %% [markdown]
# ## Sum of Purchases by Network

# %%
purchases_by_network = purchase_data_transpose.sum(axis=0)
purchases_by_network = purchases_by_network.to_frame()
purchases_by_network = purchases_by_network.rename(columns={0:'Purchases'})

# %%
# purchases_by_network.shape

# %%
# purchases_by_network.head()

# %% [markdown]
# ## Joining Purchases to Lookup Data

# %%
purchases_by_network_w_lookup = lookup_data.merge(right=purchases_by_network, left_on='Exit Survey', right_on='Source', how='left')
purchases_by_network_w_lookup.set_index('Exit Survey', inplace=True)

# %%
# purchases_by_network_w_lookup.shape

# %%
# purchases_by_network_w_lookup.head()

# %% [markdown]
# ## Spend and Lift by Network

# %%
spend_and_lift_by_network = airings_data.groupby('Network')[['Spend', 'Lift']].agg('sum')

# %%
# spend_and_lift_by_network.shape

# %%
# spend_and_lift_by_network.head()

# %% [markdown]
# ## Joining Purchases/Lookup to Spend and Lift

# %%
purchases_spend_lift_by_network = purchases_by_network_w_lookup.merge(right=spend_and_lift_by_network, left_on='Airings', right_index=True, how='left')

# %%
# purchases_spend_lift_by_network.shape

# %%
# purchases_spend_lift_by_network.head()

# %% [markdown]
# ## Computing Metrics by Network

# %%
purchases_spend_lift_by_network['Conversion Rate'] = purchases_spend_lift_by_network['Purchases'] / purchases_spend_lift_by_network['Lift'] * 100

purchases_spend_lift_by_network['Cost Per Acquisition'] = purchases_spend_lift_by_network['Spend'] / purchases_spend_lift_by_network['Purchases']

purchases_spend_lift_by_network['Cost Per Visitor'] = purchases_spend_lift_by_network['Spend'] / purchases_spend_lift_by_network['Lift']

purchases_spend_lift_by_network['Percent of Purchases'] = purchases_spend_lift_by_network['Purchases'] / sum(purchases_spend_lift_by_network['Purchases'].fillna(0)) * 100

purchases_spend_lift_by_network['Percent of Spend'] = purchases_spend_lift_by_network['Spend'] / sum(purchases_spend_lift_by_network['Spend'].fillna(0)) * 100

purchases_spend_lift_by_network['Percent Pur > Percent Spend'] = purchases_spend_lift_by_network['Percent of Purchases'] > purchases_spend_lift_by_network['Percent of Spend']

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
# # Grouped Metrics by Network and Month

# %% [markdown]
# ## Purchase Data by Network and Month

# %%
# GroupBy month to find purchases per month for each network
purchase_data_by_month = purchase_data_transpose.groupby(pd.Grouper(freq='M')).agg('sum')

# %%
# purchase_data_by_month

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
# ## Joining Purchases by network and month to Lookup Data

# %%
purchases_by_month_with_lookup = lookup_data.merge(right=purchase_data_by_month, left_on='Exit Survey', right_on='Source', how='left').set_index(['Exit Survey', 'date'])

# %%
# purchases_by_month_with_lookup.head()

# %%
# print(purchases_by_month_with_lookup.head().to_string())

# %%
# purchases_by_month_with_lookup.shape

# %% [markdown]
# ## Spend and Lift by Network and Month

# %%
spend_lift_by_network_month = airings_data.groupby(['Network', pd.Grouper(key='Date/Time ET', freq='M')])[['Spend', 'Lift']].agg('sum')

# %%
# spend_lift_by_network_month.head()

# %% [markdown]
# ## Joining Purchases/Lookup to Spend and Lift by Network and Month

# %%
purchases_spend_lift_by_network_and_month = purchases_by_month_with_lookup.reset_index().merge(right=spend_lift_by_network_month.reset_index(), left_on=['Airings', 'date'], right_on=['Network', 'Date/Time ET'], how='left')

# %%
# purchases_spend_lift_by_network_and_month.head()

# %%
# purchases_spend_lift_by_network_and_month.shape

# %%
purchases_spend_lift_by_network_and_month = purchases_spend_lift_by_network_and_month.set_index(['Exit Survey', 'date']).drop(labels=['Airings', 'Network', 'Date/Time ET', 'Source'], axis=1)

# %%
# purchases_spend_lift_by_network_and_month.head()

# %%
# purchases_spend_lift_by_network_and_month.shape

# %%
# print(purchases_spend_lift_by_network_and_month.head().to_string())

# %%
purchases_spend_lift_by_network_and_month['Conversion Rate'] = purchases_spend_lift_by_network_and_month['Purchases'] / purchases_spend_lift_by_network_and_month['Lift'] * 100

purchases_spend_lift_by_network_and_month['Cost Per Acquisition'] = purchases_spend_lift_by_network_and_month['Spend'] / purchases_spend_lift_by_network_and_month['Purchases']

purchases_spend_lift_by_network_and_month['Cost Per Visitor'] = purchases_spend_lift_by_network_and_month['Spend'] / purchases_spend_lift_by_network_and_month['Lift']

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
