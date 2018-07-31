from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("CustomerOrders")
sc = SparkContext(conf = conf)

def parseLine(line):
	fields = line.split(",")
	return (int(fields[0]), int(float(fields[2]) * 100))

input = sc.textFile("customer-orders.csv")
parsedInput = input.map(parseLine)

customerOrdersSum = parsedInput.reduceByKey(lambda x, y: x + y)
customerOrdersSorted = customerOrdersSum.map(lambda x: (x[1], x[0])).sortByKey()
customerOrdersSumMap = customerOrdersSorted.collect()

for value, customer in customerOrdersSumMap:
	print(str(customer) + "," + str(value / 100.0))