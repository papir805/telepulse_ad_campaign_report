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
import seaborn as sns
import matplotlib.pyplot as plt

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

# %% [markdown]
# ## Spend and Lift by Network

# %%
spend_and_lift_by_network = airings_data.groupby('Network')[['Spend', 'Lift']].agg('sum')

# %% [markdown]
# ## Joins

# %% [markdown]
# ### Joining Purchases by Network to Lookup Data

# %%
purchases_by_network_w_lookup = lookup_data.merge(right=purchases_by_network, left_on='Exit Survey', right_on='Source', how='left')
purchases_by_network_w_lookup.set_index('Exit Survey', inplace=True)

# %% [markdown] tags=[]
# ### Joining Purchases/Lookup by Network to Spend and Lift

# %%
purchases_spend_lift_by_network = purchases_by_network_w_lookup.merge(right=spend_and_lift_by_network, left_on='Airings', right_index=True, how='left')

# Since this column was only needed for the join, I'm going to drop it post join
purchases_spend_lift_by_network.drop('Airings', axis=1, inplace=True)

# %%
purchases_spend_lift_by_network.index = purchases_spend_lift_by_network.index.str.replace('_', ' ').str.title()

# %%
purchases_spend_lift_by_network.fillna(0, inplace=True)

# %% [markdown]
# ## Computing Metrics by Network

# %%
purchases_spend_lift_by_network['Conversion Rate (Purchases/Lift)%'] = purchases_spend_lift_by_network['Purchases'] / purchases_spend_lift_by_network['Lift'] * 100

purchases_spend_lift_by_network['Cost Per Acquisition (Spend/Purchases)'] = purchases_spend_lift_by_network['Spend'] / purchases_spend_lift_by_network['Purchases'].fillna(0)

purchases_spend_lift_by_network['Cost Per Visitor (Spend/Lift)'] = purchases_spend_lift_by_network['Spend'] / purchases_spend_lift_by_network['Lift'].fillna(0)

purchases_spend_lift_by_network['Percent of Purchases'] = purchases_spend_lift_by_network['Purchases'] / sum(purchases_spend_lift_by_network['Purchases'].fillna(0)) * 100

purchases_spend_lift_by_network['Percent of Spend'] = purchases_spend_lift_by_network['Spend'] / sum(purchases_spend_lift_by_network['Spend'].fillna(0)) * 100

purchases_spend_lift_by_network['Percent Pur > Percent Spend'] = purchases_spend_lift_by_network['Percent of Purchases'] > purchases_spend_lift_by_network['Percent of Spend']

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
# ## Creating a lookup table that has monthly date information through a Cross Join

# %% tags=[]
# First we generate a series that has the monthly date information as the index, which we can grab
month_stamps = purchase_data_transpose.groupby(pd.Grouper(freq='M')).sum().index.values

# Then we convert the series to a DataFrame and add a key column, which will be used in the cross join.  Pandas doesn't have built in cross join functionality, so this will be used as a work around
month_df = pd.DataFrame(data=month_stamps)
month_df['key']=0

# If we also add the same key to our lookup data, we'll be able to join the months in our spreadsheet to each row of the lookup table, and merge them together on that key, effectively creating a cross join
lookup_data_with_key = lookup_data.copy()
lookup_data_with_key['key'] = 0


lookup_data_with_months = lookup_data_with_key.merge(month_df)
lookup_data_with_months.rename(columns={0:'date'}, inplace=True)
lookup_data_with_months.drop(columns='key', inplace=True)

# Now we have a lookup table that has the appropriate dates for each month in the campaign for each channel

# %% [markdown]
# ## Aggregating Spend and Lift by Network and Month

# %%
spend_lift_by_network_and_month = airings_data.groupby(['Network', pd.Grouper(key='Date/Time ET', freq='M')]).sum().reset_index()

# %% [markdown]
# ## Aggregating Purchases by Network and Month

# %%
purchases_by_network_and_month = purchase_data_transpose.groupby(pd.Grouper(freq='M')).sum().transpose().stack().to_frame().reset_index()

purchases_by_network_and_month.rename(columns={0:'Purchases'}, inplace=True)

# %% [markdown]
# ## Joins

# %% [markdown]
# ### Joining lookup_data_with_months to spend_lift_by_network_and_month

# %%
lookup_spend_lift_by_network_and_month = lookup_data_with_months.merge(spend_lift_by_network_and_month, left_on=['Airings', 'date'], right_on=['Network', 'Date/Time ET'], how='left')

lookup_spend_lift_by_network_and_month.drop(columns=['Airings', 'Network', 'Date/Time ET'], inplace=True)

# %% [markdown]
# ### Joining Spend and Lift to Purchases

# %%
purchases_spend_lift_by_network_and_month = lookup_spend_lift_by_network_and_month.merge(purchases_by_network_and_month, left_on=['Exit Survey', 'date'], right_on=['Source', 'date'], how='left')

purchases_spend_lift_by_network_and_month.drop(columns='Source', inplace=True)

# %% [markdown]
# ## Cleanup

# %%
purchases_spend_lift_by_network_and_month['Exit Survey'] = purchases_spend_lift_by_network_and_month['Exit Survey'].str.replace('_', ' ').str.title()

# %%
purchases_spend_lift_by_network_and_month.rename(columns={"Exit Survey": "Exit Survey Source"}, inplace=True)

# %%
purchases_spend_lift_by_network_and_month = purchases_spend_lift_by_network_and_month.set_index(['Exit Survey Source', 'date'])

# %%
purchases_spend_lift_by_network_and_month.fillna(0, inplace=True)

# %%
purchases_spend_lift_by_network_and_month = purchases_spend_lift_by_network_and_month[['Purchases', 'Spend', 'Lift']]

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
purchases_spend_lift_by_network_and_month = purchases_spend_lift_by_network_and_month.round({"Purchases":0, "Spend":2, "Lift":0, "Conversion Rate (Purchases/Lift)%":1, "Cost Per Acquisition (Spend/Purchases)":2, "Cost Per Visitor (Spend/Lift)":2})

purchases_spend_lift_by_network_and_month[['Purchases', 'Lift']] = purchases_spend_lift_by_network_and_month[['Purchases', 'Lift']].astype(int)

purchases_spend_lift_by_network_and_month = purchases_spend_lift_by_network_and_month.sort_values('Exit Survey Source')

# %% [markdown]
# ## Output results to CSV file

# %%
purchases_spend_lift_by_network_and_month.to_csv(F"./cleaned_output/purchases_spend_lift_by_network_and_month_{current_year_and_months}.csv")

# %% [markdown]
# ## Done

# %% [markdown]
# # Generating Reports

# %% [markdown]
# ## Overall report by network

# %% tags=[]
report_for_client = purchases_spend_lift_by_network.drop(['Percent of Purchases', 'Percent of Spend', 'Percent Pur > Percent Spend'], axis=1)

report_for_client.query('Spend > 0', inplace=True)

# %%
report_for_client[['Purchases', 'Lift']] = report_for_client[['Purchases', 'Lift']].astype(int)

report_for_client = report_for_client.round({"Purchases":0, "Spend":2, "Lift":0, "Conversion Rate (Purchases/Lift)%":1, "Cost Per Acquisition (Spend/Purchases)":2, "Cost Per Visitor (Spend/Lift)":2})


report_for_client.rename_axis('Exit Survey Source', axis=0, inplace=True)

report_for_client = report_for_client.sort_values('Exit Survey Source')

# %% [markdown]
# ## Monthly report by network

# %%
report_for_client_by_month = purchases_spend_lift_by_network_and_month.drop(['Percent of Purchases', 'Percent of Spend', 'Percent Pur > Percent Spend'], axis=1)

# %%
report_for_client_by_month = report_for_client_by_month.round({"Purchases":0, "Spend":2, "Lift":0, "Conversion Rate (Purchases/Lift)%":1, "Cost Per Acquisition (Spend/Purchases)":2, "Cost Per Visitor (Spend/Lift)":2})

report_for_client_by_month[['Purchases', 'Lift']] = report_for_client_by_month[['Purchases', 'Lift']].astype(int)

# %%
# This will ensure that both reports have the same channels.  Since we already filtered report_for_client to show only channels where there was spend, report_for_client_by_month will also also have those same channels.  
report_for_client_by_month = report_for_client_by_month.loc[report_for_client.index]

# %%
report_for_client_by_month.fillna(0, inplace=True)

# %% [markdown]
# # Viewing Report by Network

# %%
report_for_client

# %% [markdown]
# # Viewing Report by Network and Month

# %%
report_for_client_by_month

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
# # Presentation 

# %% [markdown]
# ## How much does it cost to acquire a customer through TV?
# * Overall Cost Per Acquisition
# * Overall Cost Per Visitor
# * Overall Conversion Rate

# %%
# Where spend > 0
total_spend = report_for_client['Spend'].sum()
total_purchases_from_spend = report_for_client['Purchases'].sum()
total_lift = report_for_client['Lift'].sum()

overall_cost_per_acquisition = total_spend / total_purchases_from_spend
overall_cost_per_visitor = total_spend / total_lift
overall_conversion_rate = total_purchases_from_spend / total_lift * 100

# Any purchases, including where spend = 0
total_purchases_from_campaign = purchase_data_transpose.sum().sum()

cost_per_acquisition_any_spend = total_spend / total_purchases_from_campaign
conversion_rate_any_spend = total_purchases_from_campaign / total_lift * 100

print("If we only consider purchases from channels where spend > 0")
print('-'*60)
print(F"The overall cost per acquisition was: ${overall_cost_per_acquisition:.2f}")
print(F"The overall cost per visitor was: ${overall_cost_per_visitor:.2f}")
print(F"The overall conversion rate was: {overall_conversion_rate:.1f}%")
print()
print()
print("If we consider all purchases from channels, even if spend = 0")
print('-'*60)
print(F"The overall cost per acquisition was: ${cost_per_acquisition_any_spend:.2f}")
print(F"The overall conversion rate was: {conversion_rate_any_spend:.1f}%")


# %% [markdown]
# ## Cost Efficiency Metrics

# %% [markdown]
# ### Heatmaps

# %% [markdown]
# #### Plotting Function - make_heatmap()

# %%
def make_heatmap(df, field, color_map, top_labels, bottom_labels, rounding=".0f", cutoff_value=False, asc=False, annotate_horizontal=True, hide_y_label=False):

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # # Get cutoff_value isn't supplied, we use the mean as the cutoff_value
    if cutoff_value==False:
        cutoff_value = df[field].mean()
    
    
    # Sort purchases High to Low
    sorted_values = df[[field]].sort_values(by=field, ascending=asc)
    
    # Find index of channel that has a field value >= field_mean.  This is used to draw horizontal line in heatmap later.
    if asc == False:
        horizontal_y_val = sorted_values[sorted_values[field] > cutoff_value].shape[0]
    else:
        horizontal_y_val = sorted_values[sorted_values[field] < cutoff_value].shape[0]

    # Create labels for y_ticks.  Keep top and bottom 5 labels, replace middle labels
    # with empty string.
    labels_list = []
    for count, label in enumerate(sorted_values.index):
        #print(count, label)
        if count<5 or count>len(sorted_values.index)-6:
            labels_list.append(label)
        else:
            labels_list.append('')
    
    # # This part is needed if you want to get the values of Purchases to change the annotations (Annot) in the heatmap.  This keeps the top and bottom 5 values, but replaces the middle values with np.nan.  You can pass top_and_bottom_values to the sns.heatmap Annot arg to only annotate the top and bottom 5 values.

    # top_and_bottom_values = []
    # for i, boolean in enumerate(top_and_bottom_mask[field]):
    #     if boolean == True:
    #         purchases = sorted_values.iloc[i, 0]
    #         top_and_bottom_values.append(purchases)
    #         top_and_bottom_labels.append(top_and_bottom_mask.index.values[i])
    #     else:
    #         top_and_bottom_values.append(np.nan)

    
    # Unfortunately we need to hard code the names of the channels that are among the top and bottom 5 of Purchases, Spend, and Lift, b/c grabbing them programmatically is a little hard
    top_5_channels = top_labels
    bottom_5_channels = bottom_labels


    fig, ax = plt.subplots(1,1,figsize=(2,10))


    ## the last two entries for Cost Per Acquisition are np.inf and can't be plotted, so we remove them
    if field == 'Cost Per Acquisition (Spend/Purchases)':
        sorted_values = df[[field]].sort_values(by=field, ascending=True)[:-2]
    
    # Create masks
    if asc==False:
        
        mask1 = sorted_values>=sorted_values[field][4]
        ## Bottom 5 mask
        mask2 = sorted_values<=sorted_values[field][-5]
    else:
        ## Top 5 mask
        mask1 = sorted_values>=sorted_values[field][-5]
        ## Bottom 5 mask
        mask2 = sorted_values<=sorted_values[field][4]

    top_and_bottom_mask = mask1 | mask2
    middle_mask = ~top_and_bottom_mask
    
    #print(middle_mask)
    
    sns.heatmap(data=sorted_values,
                mask=middle_mask,
                annot=True, 
                cmap=color_map, 
                fmt='g',
                cbar=True,
                #annot_kws={"weight": "bold"},
                #yticklabels=purchase_labels_w_alert,
                ax=ax);


    sns.heatmap(data=sorted_values,
                # mask=top_and_bottom_mask,
                #annot=True, 
                cmap=color_map, 
                fmt='g',
                cbar = False,
                yticklabels=labels_list,
                ax=ax)

    yticks=plt.gca().get_yticklabels()

    for text in yticks:
        if text.get_text() in top_5_channels:
            text.set_weight('bold')
            text.set_color('green')
            #print('\u26A0 ' + text.get_text())
            #text.set_text('\u26A0 ' + text.get_text())
        if text.get_text() in bottom_5_channels:
            text.set_weight('bold')
            text.set_color('red')
            
    rewrite_txt_dict = {"Cost Per Visitor (Spend/Lift)":"Cost Per Visitor",
                        "Cost Per Acquisition (Spend/Purchases)":"Cost Per Acquisition",
                        "Conversion Rate (Purchases/Lift)%":"Conversion Rate"}        
    
    if field in rewrite_txt_dict.keys():
        field = rewrite_txt_dict[field]
    
    y=plt.gca().get_yticks()
    ax.tick_params(axis='y', left=False)
    
    if annotate_horizontal==True:
        ax.axhline(horizontal_y_val, linestyle=':', color='blue')
        text = ax.annotate(text=F'Avg {field} {format(cutoff_value, rounding)}:', xy=(0, horizontal_y_val), xytext=(-10, -5), textcoords='offset pixels', ha='right', color='#2596be')
        # import matplotlib.patheffects as pe
        # text.set_path_effects(path_effects=[pe.withStroke(linewidth=0.8, foreground='black'), pe.Normal()])

    if hide_y_label:
        plt.ylabel('')
    plt.show();


# %% [markdown]
# ##### Attempt at make_heatmap2() and make_multiple_heatmaps() to plot multiple heatmaps at once

# %%
def make_heatmap2(df, field, color_map, ax_to_plot_on, hide_y_label=False):

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get mean of values
    field_mean = df[field].mean()
    
    # Sort purchases High to Low
    sorted_values = df[[field]].sort_values(by=field, ascending=False)
    
    # Find index of channel that has a field value >= field_mean.  This is used to draw horizontal line in heatmap later.
    horizontal_y_val = sorted_values[sorted_values[field] > field_mean].shape[0]
    
    # Create masks

    ## Top 5 purchases mask
    mask1 = sorted_values>=sorted_values[field][4]
    ## Bottom 5 purchases mask
    mask2 = sorted_values<=sorted_values[field][-5]

    top_and_bottom_mask = mask1 | mask2
    middle_mask = ~top_and_bottom_mask

    # Create labels for y_ticks.  Keep top and bottom 5 labels, replace middle labels
    # with empty string.
    labels_list = []
    for count, label in enumerate(sorted_values.index):
        #print(count, label)
        if count<5 or count>len(sorted_values.index)-6:
            labels_list.append(label)
        else:
            labels_list.append('')

    # # This part is needed if you want to get the values of Purchases to change the annotations (Annot) in the heatmap.  This keeps the top and bottom 5 values, but replaces the middle values with np.nan.  You can pass top_and_bottom_values to the sns.heatmap Annot arg to only annotate the top and bottom 5 values.

    # top_and_bottom_values = []
    # for i, boolean in enumerate(top_and_bottom_mask[field]):
    #     if boolean == True:
    #         purchases = sorted_values.iloc[i, 0]
    #         top_and_bottom_values.append(purchases)
    #         top_and_bottom_labels.append(top_and_bottom_mask.index.values[i])
    #     else:
    #         top_and_bottom_values.append(np.nan)

    
    # Unfortunately we need to hard code the names of the channels that are among the top and bottom 5 of Purchases, Spend, and Lift, b/c grabbing them programmatically is a little hard
    top_5_channels = ['Willow Tv', 'One America News Network', 'Zeetv', 'Cnn', 'Msnbc']
    bottom_5_channels = ['Fox Sports', 'Bloomberg', 'Comedy Central', 'Turner Network Tv', 'Cnbc World']


    #fig, ax = plt.subplots(1,1,figsize=(2,10))

    sns.heatmap(data=sorted_values,
                mask=middle_mask,
                annot=True, 
                cmap=color_map, 
                fmt='g',
                cbar=True,
                #annot_kws={"weight": "bold"},
                #yticklabels=purchase_labels_w_alert,
                ax=ax_to_plot_on);


    sns.heatmap(data=sorted_values,
                # mask=top_and_bottom_mask,
                #annot=True, 
                cmap=color_map, 
                fmt='g',
                cbar = False,
                yticklabels=labels_list,
                ax=ax_to_plot_on)

    yticks=ax_to_plot_on.get_yticklabels()

    for text in yticks:
        if text.get_text() in top_5_channels:
            text.set_weight('bold')
            text.set_color('green')
            #print('\u26A0 ' + text.get_text())
            #text.set_text('\u26A0 ' + text.get_text())
        if text.get_text() in bottom_5_channels:
            text.set_weight('bold')
            text.set_color('red')

    #y=plt.gca().get_yticks()
    ax_to_plot_on.tick_params(axis='y', left=False)
    ax_to_plot_on.axhline(horizontal_y_val, linestyle=':', color='blue')
    ax_to_plot_on.annotate(text=F'Avg {field} {field_mean:.0f}:', xy=(0, horizontal_y_val), xytext=(-10, -5), textcoords='offset pixels', ha='right', color='blue')
    
    if hide_y_label:
        plt.ylabel('')
    plt.show();

# %%
fig, ax = plt.subplots(1,2)
make_heatmap2(df=report_for_client, field='Purchases', ax_to_plot_on=ax[0], color_map='Greys')
make_heatmap2(df=report_for_client, field='Purchases', ax_to_plot_on=ax[1], color_map='Greys')


# %%
def make_multiple_heatmaps(i, j, df, field, color_map, hide_y_label=False):
    fig, ax = plt.subplots(i, j, figsize=(2,10))
    make_heatmap2(df, field, color_map, ax[0], hide_y_label)
    make_heatmap2(df, field, color_map, ax[1], hide_y_label)


# %%
make_multiple_heatmaps(1, 2, report_for_client, 'Purchases', 'Blues')

# %% [markdown]
# #### Purchases, Spend, and Lift

# %%
# Top Purchases, Spend, and Lift labels
field1_top_labels = set(report_for_client['Purchases'].sort_values(ascending=False).index.values[0:5])
field2_top_labels = set(report_for_client['Spend'].sort_values(ascending=False).index.values[0:5])
field3_top_labels = set(report_for_client['Lift'].sort_values(ascending=False).index.values[0:5])

## Use set logic to find which channels are in at least 2/3 of the top5 for Purchases, Spend, and Lift
set1 = field1_top_labels.intersection(field2_top_labels)
set2 = field1_top_labels.intersection(field3_top_labels)
set3 = field2_top_labels.intersection(field3_top_labels)
set4 = set1.intersection(set2, set3)
at_least_top_2_of_3_spend_purchase_lift_labels = set1.union(set2, set3, set4)


# Bottom Purchases, Spend, and Lift labels
field1_bottom_labels = set(report_for_client['Purchases'].sort_values(ascending=False).index.values[-5:])
field2_bottom_labels = set(report_for_client['Spend'].sort_values(ascending=False).index.values[-5:])
field3_bottom_labels = set(report_for_client['Lift'].sort_values(ascending=False).index.values[-5:])

## Use set logic to find which channels are in at least 2/3 of the bottom5 for Purchases, Spend, and Lift
set1 = field1_bottom_labels.intersection(field2_bottom_labels)
set2 = field1_bottom_labels.intersection(field3_bottom_labels)
set3 = field2_bottom_labels.intersection(field3_bottom_labels)
set4 = set1.intersection(set2, set3)
at_least_bottom_2_of_3_spend_purchase_lift_labels = set1.union(set2, set3, set4)


make_heatmap(df=report_for_client, 
             field='Purchases', 
             color_map='Greys', 
             top_labels=at_least_top_2_of_3_spend_purchase_lift_labels, 
             bottom_labels=at_least_bottom_2_of_3_spend_purchase_lift_labels)
plt.show()


make_heatmap(df=report_for_client, 
             field='Spend', 
             color_map='Greys', 
             top_labels=at_least_top_2_of_3_spend_purchase_lift_labels, 
             bottom_labels=at_least_bottom_2_of_3_spend_purchase_lift_labels,
             hide_y_label=True)
plt.show()


make_heatmap(df=report_for_client, 
             field='Lift', 
             color_map='Greys', 
             top_labels=at_least_top_2_of_3_spend_purchase_lift_labels, 
             bottom_labels=at_least_bottom_2_of_3_spend_purchase_lift_labels, 
             hide_y_label=True)
plt.show()

# %% [markdown]
# #### Purchases and Cost Per Visitor

# %%
# Use set logic to find which channels are in the top5 for purchases and cost per visitor
top_5_purchases=set(report_for_client['Purchases'].sort_values(ascending=False).index.values[0:5])
top_5_cost_per_visitor=set(report_for_client['Cost Per Visitor (Spend/Lift)'].sort_values(ascending=True).index.values[0:5])

top_5_purchases_and_cost_per_visitor = top_5_purchases.intersection(top_5_cost_per_visitor)

# Use set logic to find which channels are in the bottom5 for purchases and cost per visitor
bottom_5_purchases=set(report_for_client['Purchases'].sort_values(ascending=False).index.values[-5:])
bottom_5_cost_per_visitor=set(report_for_client['Cost Per Visitor (Spend/Lift)'].sort_values(ascending=True).index.values[-5:])

bottom_5_purchases_and_cost_per_visitor = bottom_5_purchases.intersection(bottom_5_cost_per_visitor)



make_heatmap(df=report_for_client, 
             field='Purchases', 
             color_map='Greys', 
             top_labels=top_5_purchases_and_cost_per_visitor, 
             bottom_labels=bottom_5_purchases_and_cost_per_visitor, 
             annotate_horizontal=False)
plt.show()

make_heatmap(df=report_for_client, 
             field='Cost Per Visitor (Spend/Lift)', 
             cutoff_value = overall_cost_per_visitor, 
             asc=True, 
             rounding=".2f",
             top_labels=top_5_purchases_and_cost_per_visitor, 
             bottom_labels=bottom_5_purchases_and_cost_per_visitor, 
             color_map='Greys', 
             hide_y_label=True)
plt.show()

# %% [markdown]
# #### Purchases and Cost Per Acquisition

# %%
top_5_purchases=set(report_for_client['Purchases'].sort_values(ascending=False).index.values[0:5])
top_5_cost_per_acquisition=set(report_for_client['Cost Per Acquisition (Spend/Purchases)'].sort_values(ascending=True).index.values[0:5])

top_5_purchases_and_cost_per_acquisition = top_5_purchases.intersection(top_5_cost_per_acquisition)


bottom_5_purchases=set(report_for_client['Purchases'].sort_values(ascending=False).index.values[-5:])
bottom_5_cost_per_acquisition=set(report_for_client['Cost Per Acquisition (Spend/Purchases)'].sort_values(ascending=True).index.values[-5:])

bottom_5_purchases_and_cost_per_acquisition = bottom_5_purchases.intersection(bottom_5_cost_per_acquisition)

make_heatmap(df=report_for_client, 
             field='Purchases', 
             color_map='Greys', 
             top_labels=top_5_purchases_and_cost_per_acquisition, 
             bottom_labels=bottom_5_purchases_and_cost_per_acquisition, 
             annotate_horizontal=False)
plt.show()

make_heatmap(df=report_for_client, 
             field='Cost Per Acquisition (Spend/Purchases)', 
             cutoff_value = overall_cost_per_acquisition,
             top_labels = top_5_purchases_and_cost_per_acquisition,
             bottom_labels = bottom_5_purchases_and_cost_per_acquisition,
             asc=True, 
             rounding=".02f",
             color_map='Greys', 
             hide_y_label=True)
plt.show()

# %% [markdown]
# #### Purchases and Conversion Rate

# %%
top_5_purchases=set(report_for_client['Purchases'].sort_values(ascending=False).index.values[0:5])
top_5_conversion_rate=set(report_for_client['Conversion Rate (Purchases/Lift)%'].sort_values(ascending=False).index.values[0:5])

top_5_purchases_and_conversion_rate = top_5_purchases.intersection(top_5_conversion_rate)


bottom_5_purchases=set(report_for_client['Purchases'].sort_values(ascending=False).index.values[-5:])
bottom_5_conversion_rate=set(report_for_client['Conversion Rate (Purchases/Lift)%'].sort_values(ascending=False).index.values[-5:])

bottom_5_purchases_and_conversion_rate = bottom_5_purchases.intersection(bottom_5_conversion_rate)

make_heatmap(df=report_for_client, 
             field='Purchases', 
             color_map='Greys', 
             top_labels=top_5_purchases_and_conversion_rate,
             bottom_labels=bottom_5_purchases_and_conversion_rate, 
             annotate_horizontal=False)
plt.show()

make_heatmap(df=report_for_client, 
             field='Conversion Rate (Purchases/Lift)%', 
             cutoff_value = overall_conversion_rate, 
             rounding=".1f",
             top_labels = top_5_purchases_and_conversion_rate,
             bottom_labels = bottom_5_purchases_and_conversion_rate,
             color_map='Greys', 
             hide_y_label=True)
plt.show()


# %%

# %%

# %% [markdown]
# ## Scatter Plots

# %% [markdown] tags=[]
# ### Plotting Function - make_scatter()

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
    df[y_field] = df[y_field].replace(np.inf, 0)
    
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
            force_text=(1,1),
            force_points=(1,1),
            force_objects=(1,1),
            only_move={'points':'y', 'texts':'y'},
            arrowprops=dict(arrowstyle="->", color='k', lw=0.5))

    return fig, ax;


# %%
def make_scatter_with_size_adjustment(df,
                  x_field,
                  y_field,
                  size_scale,
                  x_units='',
                  y_units='',
                  color_1='green',
                  color_2='red',
                  expand_text=(1.5, 1.5),
                  expand_points=(3,3),
                  expand_objects=(3,3),
                  force_text=(1,1),
                  force_points=(1,1),
                  force_objects=(1,1)):
    
    from adjustText import adjust_text
    
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    
    # ax.tick_params(top=False,
    #                bottom=True,
    #                left=True,
    #                right=False,
    #                labelleft=False,
    #                labelbottom=False)
    
    df[x_field] = df[x_field].replace(np.inf, 0)
    df[y_field] = df[y_field].replace(np.inf, 0)
    
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
        size1 = low_x_high_y['Purchases'][i]
        if size1 == 0:
            size1 = size_scale * 1 / size_scale
        color1 = color_1
        together.append((txt1, x_coord1, y_coord1, color1))
        ax.scatter(x_coord1, y_coord1, color=color1, s=size1*size_scale)

    for i in range(len(high_x_low_y)):
        txt2 = high_x_low_y.index[i]
        x_coord2 = high_x_low_y[x_field][i]
        y_coord2 = high_x_low_y[y_field][i]
        size2 = high_x_low_y['Purchases'][i]
        if size2 == 0:
            size2 = size_scale * 1 / size_scale
        color2 = color_2
        together.append((txt2, x_coord2, y_coord2, color2))
        ax.scatter(x_coord2, y_coord2, color=color2, s=size2*size_scale)
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
            expand_text=expand_text,
            expand_points=expand_points,
            expand_objects=expand_objects,
            force_text=force_text,
            force_points=force_points,
            force_objects=force_objects,
            only_move={'points':'y', 'texts':'y'},
            arrowprops=dict(arrowstyle="->", color='k', lw=0.5))

    return fig, ax;

# %%
scale=10

# %% [markdown]
# #### Purchases vs. Spend

# %%
willow_tv = report_for_client.query("`Exit Survey Source` == 'Willow Tv'")
willow_tv_purchases = willow_tv['Purchases']
willow_tv_spend = willow_tv['Spend']
willow_tv_lift = willow_tv['Lift']

# %%
fig, ax = make_scatter(df=report_for_client,
                       x_field='Purchases',
                       y_field='Spend', 
                       #size=False,x_units='', 
                       y_units='$', 
                       color_1='red',
                       color_2='green')
ax.scatter(willow_tv_purchases, willow_tv_spend, c='blue')
text1 = ax.annotate(text='Willow Tv',
            xy=(willow_tv_purchases, willow_tv_spend),
            xytext=(-5, -20), textcoords='offset pixels',
            ha='right',
            color='blue',
            alpha=0.3)
# import matplotlib.patheffects as pe
# text1.set_path_effects(path_effects=[pe.withStroke(linewidth=2, foreground='black'), pe.Normal()])

# Label x and y values of outlier on xticks and yticks
current_xticks = ax.get_xticks()
updated_xticks = np.append(current_xticks, willow_tv['Purchases'])
ax.set_xticks(updated_xticks)

current_yticks = ax.get_yticks()
updated_yticks = np.append(current_yticks, willow_tv['Spend'])
ax.set_yticks(updated_yticks)

plt.show()

# %% [markdown]
# #### Lift vs. Purchases

# %%
fig, ax = make_scatter(report_for_client,
                       x_field='Lift',
                       y_field='Purchases',
                       x_units='',
                       y_units='')

ax.scatter(willow_tv_lift, willow_tv_purchases, c='blue')
text1 = ax.annotate(text='Willow Tv',
            xy=(willow_tv_lift, willow_tv_purchases),
            xytext=(-5, -20), textcoords='offset pixels',
            ha='right',
            color='blue',
            alpha=0.3)

# Label x and y values of outlier on xticks and yticks
current_xticks = ax.get_xticks()
updated_xticks = np.append(current_xticks, willow_tv['Lift'])
ax.set_xticks(updated_xticks)

current_yticks = ax.get_yticks()
updated_yticks = np.append(current_yticks, willow_tv['Purchases'])
ax.set_yticks(updated_yticks)

plt.show()

# %% [markdown]
# #### Lift vs. Spend

# %%
fig, ax = make_scatter_with_size_adjustment(report_for_client,
                                            x_field='Lift',
                                            y_field='Spend',
                                            size_scale=scale,
                                            x_units='',
                                            y_units='$',
                                            color_1='red',
                                            color_2='green')

text1 = ax.annotate(text='Willow Tv',
            xy=(willow_tv_lift, willow_tv_spend),
            xytext=(-5, -20), textcoords='offset pixels',
            ha='right',
            color='blue',
            alpha=0.3)

current_xticks = ax.get_xticks()
updated_xticks = np.append(current_xticks, willow_tv['Lift'])
ax.set_xticks(updated_xticks)

current_yticks = ax.get_yticks()
updated_yticks = np.append(current_yticks, willow_tv['Spend'])
ax.set_yticks(updated_yticks)

plt.show()

# %% [markdown]
# #### Conversion Rate vs. Spend

# %%
make_scatter_with_size_adjustment(report_for_client,
             x_field='Conversion Rate (Purchases/Lift)%',
             y_field='Spend',
             size_scale=scale,
             x_units='%',
             y_units='$',
             color_1='red',
             color_2='green')
plt.show();

# %% [markdown]
# #### Conversion Rate vs. Cost Per Acquisition

# %%
make_scatter_with_size_adjustment(report_for_client,
             x_field='Conversion Rate (Purchases/Lift)%',
             y_field='Cost Per Acquisition (Spend/Purchases)',
             size_scale=scale,
             x_units='%',
             y_units='$',
             color_1='red',
             color_2='green')
plt.show();

# %% [markdown]
# #### Conversion Rate vs. Cost Per Visitor

# %%
make_scatter_with_size_adjustment(df=report_for_client,
             x_field="Conversion Rate (Purchases/Lift)%",
             y_field="Cost Per Visitor (Spend/Lift)",
             size_scale=scale,
             x_units="%",
             y_units="$",
             color_1='red',
             color_2='green')
plt.show();

# %% [markdown] tags=[]
# ## Bar Charts

# %% [markdown] tags=[]
# ### Channels with no spend, but had purchases.  Excluding 'Other' and '(Blank)'

# %%
no_spend_but_purchases = purchases_spend_lift_by_network.query("Spend == 0 & Purchases > 0 & `Exit Survey` != 'Other' & `Exit Survey` != '(Blank)'")
no_spend_but_purchases

# %%
no_spend_but_purchases = purchases_spend_lift_by_network.query("Spend == 0 & Purchases > 0 & `Exit Survey` != 'Other' & `Exit Survey` != '(Blank)'")

no_spend_but_purchases = no_spend_but_purchases.sort_values('Purchases', ascending=False)[['Purchases']]

# %%
no_spend_but_purchases

# %%
no_spend_but_purchases['percent_of_all_purchases'] = no_spend_but_purchases['Purchases'] / total_purchases_from_campaign * 100

# %%
mean_num_purchases_with_spend = round(report_for_client['Purchases'].mean(), 0)
mean_num_purchases_from_campaign = round(purchases_spend_lift_by_network['Purchases'].mean(), 0)

# %%
no_spend_but_above_mean_purchases_from_campaign_labels = no_spend_but_purchases[no_spend_but_purchases['Purchases'] > mean_num_purchases_from_campaign].index.values

# %%
# no_spend_but_purchases['above_mean_purchases_from_campaign'] = no_spend_but_purchases['Purchases'] > mean_num_purchases_from_campaign

# %%
no_spend_yticks = []
for label in no_spend_but_purchases.index.values:
    if label in no_spend_but_above_mean_purchases_from_campaign_labels:
        no_spend_yticks.append(label)
    else:
        no_spend_yticks.append('')

# %%
import matplotlib.patheffects as pe

fig, ax = plt.subplots(1,1,figsize=(8,6))
no_spend_but_purchases['Purchases'].plot(kind='barh', ax=ax, edgecolor='black')
ax.set_xlabel('Number of Purchases on Exit Survey')
ax.set_title('Channels where spend = 0')


ax.axvline(mean_num_purchases_with_spend, color='red', linestyle='--')
text1 = ax.annotate(F'Mean purchases\nfrom channels\nthat had spending',
                xy=(mean_num_purchases_with_spend, 0), xycoords='data',
                xytext=(5, -175), textcoords='offset pixels', size=11,
                color='red', ha='left')
# text1.set_path_effects(path_effects=[pe.withStroke(linewidth=2.5, foreground='black'), pe.Normal()])

ax.axvline(mean_num_purchases_from_campaign, color='darkviolet', linestyle='--')
text2 = ax.annotate(F'Mean purchases\noverall',
                xy=(mean_num_purchases_from_campaign, 0), xycoords='data',
                xytext=(5, -175), textcoords='offset pixels', size=11,
                color='darkviolet', ha='left')
# text2.set_path_effects(path_effects=[pe.withStroke(linewidth=1.3, foreground='black'), pe.Normal()])

ax.set_yticklabels(no_spend_yticks)

yticks=plt.gca().get_yticklabels()



for text in yticks:
    if text.get_text() in no_spend_but_above_mean_purchases_from_campaign_labels:
        text.set_weight('bold')
        text.set_color('green')

ax.invert_yaxis();

# %% [markdown]
# # Scratch Work

# %% [markdown]
# ## Which networks have no spend?

# %%
purchases_spend_lift_by_network.query('Spend == 0')

# %%

# %%
num_purchases_no_spend = report_for_client[report_for_client['Spend']==0].groupby("Exit Survey Source")['Purchases'].agg('sum')

tot_purchases_no_spend = sum(num_purchases_no_spend)

perc_purchases_no_spend = num_purchases_no_spend / tot_purchases_no_spend * 100
perc_purchases_no_spend = perc_purchases_no_spend.sort_values(ascending=False)

perc_purchases_no_spend
# (report_for_client[report_for_client['Spend']==0].groupby("Exit Survey Source")['Purchases'].agg('sum') / sum(report_for_client[report_for_client['Spend']==0].groupby("Exit Survey Source")['Purchases'].agg('sum')) * 100).sort_values(ascending=False)

# %%

# %% [markdown]
# ## Old Graphing

# %% [markdown]
# ### Bar Charts

# %%
# ax = report_purchases_sorted[0:10].plot(kind='barh', y='Purchases', title='Top 10 Networks by Purchase', legend=False)
# ax.invert_yaxis()
# ax.set_xlabel('Number of Purchases');

# %%
# ax = report_purchases_sorted[-10:].plot(kind='barh', y='Purchases', title='Bottom 10 Networks by Purchase', legend=False)
# ax.invert_yaxis()
# ax.set_xlabel('Number of Purchases')
# ax.set_xticks(np.arange(0,3,1));

# %%
# ax = report_purchases_sorted[0:10].plot(kind='barh', y='Purchases', title='Top 10 Networks by Purchase', legend=False, color='blue')
# ax.invert_yaxis()
# ax.set_xlabel('Number of Purchases')
# ax1 = ax.twiny()
# report_purchases_sorted[0:10].plot(kind='barh', y='Spend', title='Top 10 Networks by Purchase', legend=False, ax=ax, color='r');

# %%
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_xlabel('Number of Purchases')


# ax2 = ax.twiny()
# ax2.set_xlabel('Amount Spent ($)')


# bar1 = report_purchases_sorted['Purchases'][0:10].plot(kind='barh', color='green', position=1, width=0.4, ax=ax, label='Purchases')
# bar2 = report_purchases_sorted['Spend'][0:10].plot(kind='barh', color='red', position=0, width=0.4, ax=ax2, label='Spend')

# #ax2.invert_yaxis()
# plt.ylim((-0.5, len(report_purchases_sorted[0:10])-0.5))
# ax.invert_yaxis()

# bars, labels = ax.get_legend_handles_labels()
# bars2, labels2 = ax2.get_legend_handles_labels()
# ax.legend(bars+bars2, labels+labels2, loc='lower right')

# ;

# %%
# fig = plt.figure(figsize=(7,7))
# ax = fig.add_subplot(111)
# ax.set_xlabel('Number of Purchases')


# ax2 = ax.twiny()
# ax2.set_xlabel('Amount Spent ($)')


# report_purchases_sorted['Purchases'][-10:].plot(kind='barh', color='blue', position=1, width=0.4, ax=ax)
# report_purchases_sorted['Spend'][-10:].plot(kind='barh', color='red', position=0, width=0.4, ax=ax2)
# ax.invert_yaxis()
# #ax2.invert_yaxis()
# plt.ylim((-0.5, len(report_purchases_sorted[0:10])-0.5));

# %%

# %%

# %%

# %% [markdown]
# ### Scatter Plots

# %% [markdown]
# #### Adjusting annotation text manually

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
# #### Where you started using adjustText to adjust annotation text automatically

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
