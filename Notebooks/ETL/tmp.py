import pandas as pd

# # Assuming 'mock_dates' is your list of dates in various formats
# mock_dates = ['1976/08/30', 'September 24, 2013','25 September, 2013' '1986-10-05', '2012.10.01', '05/29/2017', 'March 05, 1985', '1982.09.10']


# Sample data with inconsistent date formats
data = {
    'Event': ['Concert', 'Conference', 'Meeting', 'Workshop', 'Seminar'],
    'Date': ['12/05/2024', '05/15/2024', '23/06/2024', 'September 24, 2013', '2012.10.01']
}

df = pd.DataFrame(data)

# Converting dates with errors='coerce' to handle unparseable formats
# converted_dates = pd.to_datetime(mock_dates, format='mixed', errors='coerce')
converted_dates = pd.to_datetime(df['Date'], format="mixed", errors='coerce', dayfirst=True)


print(converted_dates)