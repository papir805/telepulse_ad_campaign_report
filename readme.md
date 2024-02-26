# TelePulse Ad Campaign Report
## Goal: Automating the generation of monthly reports to determine the most cost-efficient TV networks for ad spending.
Important Python libraries used: `Pandas`, `NumPy`, `Matplotlib`, and `Seaborn`.
<br>
<br/>

![TV](./thumbnail.jpeg)

Every month, a data pipeline generates an Excel file containing information on ads shown on over 30 TV networks. Each row in the table represents data on a particular ad airing in a specific time slot, consisting of:
* Amount spent on the ad
* Lift generated from the ad
* Customer acquisitions from the ad

My Python script cleans the data, performs joins, and calculates important metrics, then generates and exports three summary reports as PDF files to produce recommendations for the TV networks where ad spending should be increased or decreased.

To download the original dataset as an Excel .xlsw file, [click here](./dataset.xlsx)

The script that cleans the data and produces the reports can be found [here](https://nbviewer.org/github/papir805/telepulse_ad_campaign_report/blob/main/ad_campaign_report_script.ipynb)
<br>
<br/>


Using data from September and October 2017, I've generated the [three reports](./output/reports/pdfs) and created visuals in another jupyter notebook to help identify the most cost efficient networks. 

The jupyter notebook that generated the visuals can be found [here](https://github.com/papir805/telepulse_ad_campaign_report/blob/main/ad_campaign_visuals.ipynb)
<br>
<br/>

Lastly, I've taken the visuals and created a PowerPoint presentation with my recommendations for the TV networks where advertising spending should be increased or decreased.

The Google Slides presentation can be found [here](https://docs.google.com/presentation/d/1T-fGZ3Cf7lJvf4lJWJhyOq45gDOGqSKuG6wpV-fVQLo/edit?usp=sharing)
<br>
<br/>



<!-- # To-Do List - ad_campaign_report_script:
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
    - [ ] Make title slides more informative by leveraging pre-attentive attributes.  The title sets the tone for how one reads the rest of the slide. -->
