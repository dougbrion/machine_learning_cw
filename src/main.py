import csv

no = 0
year = 1
month = 2
day = 3
hour = 4
pm2_5 = 5
DEWP = 6
TEMP = 7
PRES = 8
cbwd = 9
lws = 10
ls = 11
lr = 12

def load_csv(path, filename):
    raw_data = open(path + filename, "rt")
    lines = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    dataset = list(lines)
    return dataset

filename = 'PRSA_data_2010.1.1-2014.12.31.csv'
path = '../data/'
dataset = load_csv(path, filename)

numOfReadings = len(dataset)

print(len(dataset))
print(len(dataset[0]))

avgTemp = 0
avgDewp = 0
avgPres = 0

for x in range (1, numOfReadings):
    avgDewp += float(dataset[x][DEWP]) / numOfReadings
    avgTemp += float(dataset[x][TEMP]) / numOfReadings
    avgPres += float(dataset[x][PRES]) / numOfReadings

print("Average Dew Point: ", avgDewp)
print("Average Temperature: ", avgTemp)
print("Average Pressure: ", avgPres)