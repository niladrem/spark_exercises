import sys
from pyspark import SparkConf, SparkContext
from math import sqrt
from statistics import mean
import configparser

def loadMovieNames():
	movieNames = {}
	f = open('ml-100k/u.item', encoding='ascii', errors='ignore')
	for line in f:
		fields = line.split("|")
		movieNames[int(fields[0])] = fields[1]
	return movieNames

def parseLine(line):
	l = line.split()
	return (int(l[0]), (int(l[1]), float(l[2]))) # userID -> (movieID, rating)

def parseLineScore(line):
	l = line.split()
	return (int(l[1]), float(l[2]))

def makePairs(userRatings):
	ratings = userRatings[1]
	return ((ratings[0][0], ratings[1][0]), (ratings[0][1], ratings[1][1])) # ((movie1, movie2), (rating1, rating2))

def filterDuplicates(userRatings):
	ratings = userRatings[1]
	return ratings[0][0] < ratings[1][0]

def filterScores(pairScores, movieId, scoreThreshold, coOccurenceThreshold):
	movies = pairScores[0]
	scores = pairScores[1]
	return (movies[0] == movieID or movies[1] == movieID) and scores[0] >= scoreThreshold and scores[1] >= coOccurenceThreshold

def cosineSimilarity(ratingPairs):
	numPairs = 0
	xx = yy = xy = 0
	for x, y in ratingPairs:
		xx += x * x
		yy += y * y
		xy += x * y
		numPairs += 1

	score = 0
	if xx and yy:
		score = xy / (float(sqrt(xx) * sqrt(yy)))
	return (score, numPairs)

def pearsonCorrelationCoefficient(ratingPairs):
	numPairs = 0
	_x = _y = 0
	for x, y in ratingPairs:
		_x += x
		_y += y
		numPairs += 1
	_x /= numPairs
	_y /= numPairs

	xy = xx = yy = 0
	for x, y in ratingPairs:
		xy += (x - _x) * (y - _y)
		xx += (x - _x) ** 2
		yy += (y - _y) ** 2
	score = 0
	if xx and yy:
		score = xy / (sqrt(xx) * sqrt(yy))
	return (abs(score), numPairs)

def jaccardCoefficient(ratingPairs):
	numPairs = intersect = 0
	for x, y in ratingPairs:
		numPairs += 1
		if x == y:
			intersect += 1
	score = intersect / numPairs
	return (score, numPairs)

def euclideanDistance(ratingPairs):
	numPairs = 0
	score = 0
	for x, y in ratingPairs:
		score += (x - y) ** 2
		numPairs += 1
	score = score / numPairs / 36 # normalized value
	return (1 - sqrt(score), numPairs)

def metricSelector(ratingPairs, metric):
	if metric == 'COSINE':
		return cosineSimilarity(ratingPairs)
	elif metric == 'EUCLIDEAN':
		return euclideanDistance(ratingPairs)
	elif metric == 'PEARSON':
		return pearsonCorrelationCoefficient(ratingPairs)
	elif metric == 'JACCARD':
		return jaccardCoefficient(ratingPairs)
	else:
		return cosineSimilarity(ratingPairs)

conf = SparkConf().setMaster("local[*]").setAppName("MovieSimilarities")
sc = SparkContext(conf = conf)

config = configparser.ConfigParser()
config.read("config.ini")
metric = config['DEFAULT']['METRIC']

nameDict = loadMovieNames()

data = sc.textFile("ml-100k/u.data")
ratings = data.map(parseLine)

avgRating = data.map(parseLineScore).groupByKey().mapValues(mean)
avgRatingBroadcast = sc.broadcast(avgRating.collectAsMap())

joinedRatings = ratings.join(ratings).filter(filterDuplicates)
moviePairRatings = joinedRatings.map(makePairs).groupByKey()

moviePairSimilarities = moviePairRatings.mapValues(lambda row: metricSelector(row, metric)).cache()

if len(sys.argv) > 1:
	movieID = int(sys.argv[1])
	scoreThreshold = float(config[metric]['SCORE_THRESHOLD'])
	coOccurenceThreshold = int(config[metric]['COOCCURENCE_THRESHOLD'])

	# get only good movies
	minAvgScore = int(config[metric]['MINIMUM_AVG_SCORE'])
	if minAvgScore:
		moviePairSimilarities = moviePairSimilarities.filter(lambda row: row[0][0] == movieID and avgRatingBroadcast.value[row[0][1]] >= minAvgScore or row[0][1] == movieID and avgRatingBroadcast.value[row[0][0]] >= minAvgScore)

	filteredResults = moviePairSimilarities.filter(lambda row: filterScores(row, movieID, scoreThreshold, coOccurenceThreshold))

	# sort by quality score
	results = filteredResults.map(lambda pairSimilarities: (pairSimilarities[1], pairSimilarities[0])).sortByKey(ascending = False).take(10)
	
	print("Top 10 similar movies for " + nameDict[movieID])
	for result in results:
		(similarity, pair) = result
		similarMovieID = pair[0]
		if similarMovieID == movieID:
			similarMovieID = pair[1]
		print(nameDict[similarMovieID] + "\n\tscore: " + str(similarity[0]) + "\tstrength: " + str(similarity[1]))