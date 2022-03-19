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

# %%
import pandas as pd
import numpy as np
from datetime import datetime
#import itertools

# %%
purchase_data = pd.read_excel("./Analyst_dataset.xlsx", sheet_name='Purchase Exit Survey Data')
airings_data = pd.read_excel("./Analyst_dataset.xlsx", sheet_name='Airings')
lookup_data = pd.read_excel("./Analyst_dataset.xlsx", sheet_name='Lookup', skiprows=1)

# %%
purchase_data.head()

# %%
airings_data.head()

# %%
lookup_data.head()

# %% [markdown]
# # Converting dates from columns to rows in Purchase Exit Survey Data

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
# current_month = months[0]
# i = 0
# for count, day in enumerate(day_nums, start=1):
#     if str(day) > str(purchase_data.iloc[3, 2+count]):
#         i += 1
#         current_month = month[i]
#         current_date = str(current_year) + '-' + current_month + '-' + str(day)
#         print(current_date)
#         #print(count)
#     else:
#         current_date = str(current_year) + '-' + current_month + '-' + str(day)
#         print(current_date)
#         #print(count)

day_nums = np.array(purchase_data.iloc[3,2:], dtype=int)

parsed_dates = []
current_month = months[0]
i = 0
for count, today in enumerate(day_nums, start=1):
    try:
        tomorrow = day_nums[count]
    except:
        pass
        # current_date = str(current_year) + '-' + current_month + '-' + str(today)
        # current_date = datetime.strptime(current_date, '%Y-%B-%d').date()
        # parsed_dates.append(current_date)
        #print(type(current_date))
    if today > tomorrow:
        current_date = str(current_year) + '-' + current_month + '-' + str(today)
        current_date = datetime.strptime(current_date, '%Y-%B-%d').date()
        i += 1
        current_month = months[i]
        parsed_dates.append(current_date)
        #print(current_date)
        #print(count)
    else:
        current_date = str(current_year) + '-' + current_month + '-' + str(today)
        current_date = datetime.strptime(current_date, '%Y-%B-%d').date()
        #print(current_date)
        parsed_dates.append(current_date)
        #print(count)

# %%
purchase_data.iloc[3,2:] = parsed_dates

# %%
purchase_data_transpose = purchase_data.iloc[3:,:].transpose()
#new_df.head()

# %%
purchase_data_transpose.index = purchase_data_transpose.iloc[:, 0]

# %%
purchase_data_transpose = purchase_data_transpose.iloc[1:]

# %%
purchase_data_transpose = purchase_data_transpose.drop(labels=3, axis=1)

# %%
purchase_data_transpose.columns = purchase_data_transpose.iloc[0]

# %%
purchase_data_transpose = purchase_data_transpose.drop(labels='Source')

# %%
purchase_data_transpose.index = pd.to_datetime(purchase_data_transpose.index)

# %%
purchase_data_transpose.rename_axis('date', inplace=True)
#purchase_data_transpose

# %% [markdown]
# # Overall metrics by Network

# %%
sum_of_purchases = purchase_data_transpose.sum(axis=0)
sum_of_purchases

# %%
overall_tbl = lookup_data.merge(right=sum_of_purchases.to_frame(), left_on='Exit Survey', right_on='Source', how='left')
overall_tbl.drop(labels='Exit Survey.1', axis=1, inplace=True)
overall_tbl.rename(columns={0:'Purchases'}, inplace=True)
overall_tbl.set_index('Exit Survey', inplace=True)
#overall_tbl

# %%
airings_spend_and_lift = airings_data.groupby('Network')[['Spend', 'Lift']].agg('sum')
#airings_spend_and_lift

# %%
overall_tbl = overall_tbl.merge(right=airings_spend_and_lift,left_on='Airings', right_index=True, how='left')
#overall_tbl

# %%
overall_tbl['Conversion Rate'] = overall_tbl['Purchases'] / overall_tbl['Lift'] * 100
overall_tbl['Cost Per Acquisition'] = overall_tbl['Spend'] / overall_tbl['Purchases']
overall_tbl['Cost Per Visitor'] = overall_tbl['Spend'] / overall_tbl['Lift']
overall_tbl['Percent of Purchases'] = overall_tbl['Purchases'] / sum(overall_tbl['Purchases'].fillna(0)) * 100
overall_tbl['Percent of Spend'] = overall_tbl['Spend'] / sum(overall_tbl['Spend'].fillna(0)) * 100
overall_tbl['Percent Pur > Percent Spend'] = overall_tbl['Percent of Purchases'] > overall_tbl['Percent of Spend']
overall_tbl

# %%

# %%

# %%

# %% [markdown]
# # Grouped Metrics by Network and Month

# %%
purchase_data_by_date = purchase_data_transpose.groupby(pd.Grouper(freq='M')).agg('sum')
purchase_data_by_date

# %%
purchase_data_by_date = purchase_data_by_date.transpose()
purchase_data_by_date = purchase_data_by_date.stack().to_frame()
purchase_data_by_date.rename(columns={0:'Purchases'}, inplace=True)
purchase_data_by_date

# %% [markdown]
# # Airings Sheet

# %%
airings_data.info()

# %%
# airings_data.groupby([pd.Grouper(key='Date/Time ET', freq='M'), 'Network'])[['Spend', 'Lift']].agg('sum')

# %%
lookup_data = lookup_data.drop('Exit Survey.1', axis=1)
lookup_data = lookup_data.set_index('Exit Survey')
lookup_data = lookup_data.rename_axis('Source')
lookup_data

# %% [markdown]
# # Joining Purchases Exit Survey Data and Airings

# %%
# joined_tbl = lookup_data[['Exit Survey', 'Airings']].merge(right=purchase_data_by_date, left_on='Exit Survey', right_on='Source', how='left')
# #joined_tbl.drop(labels='Exit Survey', axis=1, inplace=True)
# joined_tbl

# %%
purchase_grouped = purchase_data_by_date.join(lookup_data, how='left')
purchase_grouped

# %%
# NEED TO drop Network as an index and make a column, join with purchase_grouped and keep the 
airings_spend_lift_grouped = airings_data.groupby(['Network', pd.Grouper(key='Date/Time ET', freq='M')])[['Spend', 'Lift']].agg('sum')
airings_spend_lift_grouped

# %%
airings_spend_lift_grouped.reset_index()

# %%
purchase_grouped.reset_index()

# %%
month_and_network_grouped = purchase_grouped.reset_index().merge(right=airings_spend_lift_grouped.reset_index(), left_on=['Airings', 'date'], right_on=['Network', 'Date/Time ET'], how='left')
month_and_network_grouped

# %%
month_and_network_grouped= month_and_network_grouped.set_index(['Source', 'date']).drop(labels=['Airings', 'Network', 'Date/Time ET'], axis=1)
month_and_network_grouped

# %%

# %%

# %%

# %%

# %%

# %%
airings_data.query('Network == "FOOD"')

# %%

# %% [markdown]
# # What networks have purchases but no spend?

# %%
airings_data.query('Spend == 0')['Network'].value_counts()

# %%
airings_data.groupby('Network')[['Spend', 'Lift']].agg('sum')

# %%
