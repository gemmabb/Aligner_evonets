Data from paper: Kang, S., Yuhaoand Gao, Liang, Y., Li, M., Rao, J. & Kruse, J. Multiscale dynamic human mobility flow dataset in the u.s. during the covid-19 epidemic. Sci. Data 7, 390, 10.1038/s41597-020-00734-5 (2020).

Specifically, we donwloaded the data from their GitHub repository by running the bash code in download_counties_2019_firstweeks.sh

From this, we obtained 12 datasets, one for each month in 2019 with county to county flows.
Each dataset (weekly_county2county_2019_MM_dd.csv where MM_dd correspond to the date of the first monday of each month) contains: geoid_o, geoid_d, lng_o, lat_o, lng_d, lat_d, date_range, visitor_flows, and pop_flows.