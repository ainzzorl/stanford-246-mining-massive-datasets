import sys
from pyspark import SparkConf, SparkContext

conf = SparkConf()
conf.set('spark.hadoop.validateOutputSpecs', 'false')
sc = SparkContext(conf=conf)

MAX_OUTPUT_LEN = 10

def to_first_degree_friends(line):
    parts = line.split('\t')
    return (parts[0], parts[1].split(','))

# Key: user, value: list<friend>
first_degree_friends = sc.textFile(sys.argv[1]).map(to_first_degree_friends)
# Uncomment to test on a small set
# first_degree_friends = sc.parallelize(first_degree_friends.take(1000))

# Key: (user1, user2), value: 1
# user1 and user2 are second-degree friends (may be first degree as well)
second_degree_friends_pairs = first_degree_friends.flatMap(lambda r: [((v1, v2), 1) for v1 in r[1] for v2 in r[1] if v1 != v2])

# Key: (user1, user2), value: num-common-friends
second_degree_counted = second_degree_friends_pairs.reduceByKey(lambda agg, cur: agg + cur)

# Key: user, value: list<(second-degree-friend, num-common-friends)>
candidates = second_degree_counted.map(lambda row: (row[0][0], (row[0][1], row[1]))).groupByKey()

def filter_joined_second_and_first(row):
    key = row[0]
    cds = row[1][0]
    first = row[1][1]
    return (key, [candidate for candidate in cds if candidate[0] not in first])

# Key: user, value: list<(second-degree-friend, num-common-friends)>
# First degree friends filtered out
filtered_candidates = candidates.join(first_degree_friends).map(filter_joined_second_and_first)

# Candidates are sorted, top MAX_OUTPUT_LEN are taken for each user
sorted_candidates = filtered_candidates.map(lambda row: (row[0], sorted(row[1], key=lambda candidate: (-candidate[1], int(candidate[0])))[:MAX_OUTPUT_LEN]))

sorted_candidates.saveAsTextFile(sys.argv[2])
sc.stop()
