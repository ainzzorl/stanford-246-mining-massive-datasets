import re
import sys
from pyspark import SparkConf, SparkContext
conf = SparkConf()
conf.set("spark.hadoop.validateOutputSpecs", "false")
sc = SparkContext(conf=conf)

lines = sc.textFile(sys.argv[1])

def is_relevant_word(word):
    if not word:
        return False
    first_lower_char = word[0].lower()
    return first_lower_char >= 'a' and first_lower_char <= 'z'

def get_relevant_words(line):
    ws = re.split(r'[^\w]+', line)
    return [word.lower() for word in ws if is_relevant_word(word)]

words = lines.flatMap(get_relevant_words)

pairs = words.map(lambda w: (w[0], 1))

counts = pairs.reduceByKey(lambda n1, n2: n1 + n2)

counts.saveAsTextFile(sys.argv[2])
sc.stop()
