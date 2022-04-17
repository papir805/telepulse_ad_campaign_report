# Television Advertising Campaign Report
## Goal: To produce a script, which automatically generates monthly reports, that can be used to determine the most cost efficient TV networks to advertise on.

Every month, a data pipeline generates an Excel report containing information relating to ads shown on over 30 TV networks, over the past two months, for a particular company/brand.  The data consists of amount spent on, lift generated from, and customer acquisition attributed to a particular ad over the past two months of airings.  This script cleans the data, performs joins, and calculates important metrics, then generates three and exports reports as PDF files, which can be used to produce recommendations for TV networks where ad spending should be increased/decreased.

The script can be found [here](https://github.com/papir805/ad_campaign_report/blob/main/ad_campaign_report_script.ipynb)

Using data from September and October 2017, I've generated the [three reports](https://github.com/papir805/ad_campaign_report/tree/main/output/reports/pdfs) and created visuals in another jupyter notebook to help identify the most cost efficient networks. 

The jupyter notebook that generated the visuals can be found [here](https://github.com/papir805/ad_campaign_report/blob/main/ad_campaign_visuals.ipynb)

Lastly, I've taken the visuals and created a PowerPoint presentation with my recommendations for which TV networks more/less should be spent on ads.

The PowerPoint presentation can be found [here need-to-fill-in]()

To download the original dataset as an Excel .xlsw file, [click here](https://github.com/papir805/ad_campaign_report/raw/main/dataset.xlsx)

Important Python libraries used: `Pandas`, `NumPy`, `Matplotlib`, and `Seaborn`.

# To-Do List - ad_campaign_report_script:
- None

# To-Do List - ad_campaign_visuals:
- Scatter plots
    - Spend Vs. Conversion Rate
        - [ ] Change annotations to shorter network names (ex: Turner Network Television -> TNT)
        - [ ] Learn parameters of adjust_text library better in order to get labels for points aligned properly
        - [ ] Remove unnecessary labels, not all are needed for PowerPoint presentation
    - Fix make_scatter() function.  I need to ensure that the horizontal/vertical dashed lines are the correct mean values, then apply the fix to these scatter plots:
        - [ ] Conversion Rate vs Cost Per Acquisition
        - [ ] Conversion Rate vs Cost Per Visitor
    - [ ] Consider removing scatter plot with 2 green and 1 red on border.  Does this scatter plot really add anything to the presentation?
    - [ ] Make non-essential points on scatter plots grey, to fade them to background
    - [ ] Add appropriate labels (ex: $/%) for x or y-axis values
- Heat Maps
    - [ ] Add appropriate labels (ex: $/%) for x or y-axis values
- Bar Charts
    - None
- Slope Chart
    - [ ] Incorporate to show monthly changes?
- PowerPoint presentation
    - [ ] Add agenda at start
    - [ ] Make title slides more informative by leveraging pre-attentive attributes.  The title sets the tone for how one reads the rest of the slide.
