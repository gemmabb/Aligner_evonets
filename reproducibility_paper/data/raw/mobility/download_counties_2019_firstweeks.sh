# List of first-week start dates for each month in 2019
dates=("2019_01_07" "2019_02_04" "2019_03_04" "2019_04_01" 
       "2019_05_06" "2019_06_03" "2019_07_01" "2019_08_05" 
       "2019_09_02" "2019_10_07" "2019_11_04" "2019_12_02")

# Loop through each date
for date in "${dates[@]}"; do
    # Download the CSV file directly into the current directory
    wget "https://raw.githubusercontent.com/GeoDS/COVID19USFlows-WeeklyFlows/refs/heads/master/weekly_flows/county2county/weekly_county2county_${date}.csv"
done
