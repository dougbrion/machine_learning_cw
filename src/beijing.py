from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot

path = "../data/"
raw_file = "PRSA_data_2010.1.1-2014.12.31.csv"
transformed_file = "pollution.csv"

def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')

def transform_data(path, in_file, out_file):
	dataset = read_csv(path + in_file, parse_dates = [['year', 'month', 'day', 'hour']], index_col = 0, date_parser = parse)
	dataset.drop('No', axis = 1, inplace = True)
	dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
	dataset.index.name = 'date'
	# replace all NA values in data with 0
	dataset['pollution'].fillna(0, inplace = True)
	# remove first 24 hours
	dataset = dataset[24:]
	# print out first 5 rows as example
	print(dataset.head(5))
	# save converted data to file
	dataset.to_csv(path + out_file)

def plot_data(path, in_file):
	# read in data
	dataset = read_csv(path + in_file, header = 0, index_col = 0)
	values = dataset.values
	groups = [0, 1, 2, 3, 5, 6, 7]
	currentPlot = 1
	# create pyplot
	pyplot.figure()
	# iterate through members in group creating sub plots
	for group in groups:
		pyplot.subplot(len(groups), 1, currentPlot)
		pyplot.plot(values[:,group])
		pyplot.title(dataset.columns[group], y = 0.5, loc='right')
		currentPlot += 1
	pyplot.show()

transform_data(path, raw_file, transformed_file)
plot_data(path, transformed_file)