# PySpark

## Data LifeCycle

```
Collect --> Analyze --> Clean --> Organize --> Transform --> Insight
```

- Spark, Spark Streaming, Spark ML, Spark GraphX

```
Code --> Driver Program (SparkContext) --> Cluster Manager(YARN, mesos, kubernetes, etc.) --> Worker Nodes
```

- SparkXGBoost

## Databricks

- Data Analysis and data cleaning (in databricks)

```
\# File location and type
file_location = 'LoanStats_1028Q4.csv'
file_type = 'csv'

\# CSV options
infer_schema = 'true'
first_row_is_header = 'true'
delimiter = ','

df = spark.read.format(file_type) \
	.option('inferSchema', infer_schema) \
	.option('header', first_row_is_header) \
	.option('sep', delimiter) \
	.load(file_location)

display(df)


```


















## Udemy

# Setup

- Databricks Setup
- Local Virtual Box Setup
- AWS EC2 PySpark Setup
- AWS EMR Cluster Setup

# Code

```
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('Basics').getOrCreate()

df = spark.read.json('people.json')

df.show()

df.printSchema()

df.columns

df.describe().show()
```

```
from pyspark.sql.types import StructField, StringType, IntegerType, StructType

data_schema = [StructField('age', IntegerType(), True), 
	       StructField('name', StringType(), True)]

final_struct = StructType(fields=data_schema)

df = spark.read.json('people.json', schema=final_struc)

df.printSchema()
```

```
type(df['age'])

type(df.select('age'))

df.select('age').show() # creating and displaying new dataframe

type(df.head(2)[0])

df.select(['age', 'name']).show()

df.withColumn('newage', df['age']).show() # Adding a new column to the dataframe

df.withColumn('double_age', df['age']*2).show()

df.withColumnRenamed('age', 'my_new_age').show() # Renaming the existing column

df.createOrReplaceTempView('people')

results = spark.sql("SELECT * FROM people")

results.show()

new_results = spark.sql("SELECT * FROM people WHERE age = 30")

new_results.show()
```

```
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('ops').getOrCreate()

df = spark.read.csv('appl_stock.csv', inferSchema=True, header=True)

df.printSchema()

df.show()

df.head(3)[0]

df.filter("Close < 500").show() # sql

df.filter("Close < 500").select('Open').show() # sql

df.filter(df['Close'] < 500).show() # Dataframe syntax

df.filter(df['Close'] < 500).select('Volume').show()

df.filter( (df['Close'] < 200) & (df['Open'] > 200)).show()

df.filter( (df['Close'] < 200) & ~(df['Open'] > 200)).show() # ~ not operator

df.filter(df['Low'] == 197.16).show() # Just displays the row, cannot really work with it
 
result = df.filter(df['Low'] == 197.16).collect() # gets the row that we can work with

row = result[0]

row.asDict()

row.asDict()['Volume']
```

```
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('aggs').getOrCreate()

df = spark.read.csv('sales_info.csv', inferSchema=True, header=True)

df.show()

df.printSchema()

df.groupBy("Company").mean().show()

df.groupBy('Company').count().show()

df.agg({'Sales':'max'}).show()

group_data = df.groupBy('Company')

group_data.agg({'Sales':'max'}).show()

from pyspark.sql.functions import countDistinct, avg, stddev

df.select(avg('Sales')).show()

df.select(countDistinct('Sales')).show()

df.select(avg('Sales').alias('Average Sales')).show()

df.select(stddev('Sales')).show()

from pyspark.sql.functions import format_number

sales_std = df.select(stddev('Sales').alias('std'))

sales_std.select(format_number('std',2).alias('final')).show()

df.orderBy("Sales").show()

df.orderBy(df['Sales'].desc()).show()
```

```
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('miss').getOrCreate()

df = spark.read.csv('ContainsNull.csv', inferSchema=True, header=True)

df.show()

df.na.drop().show()

df.na.drop(thresh=2).show() 

df.na.drop(how='all').show()

df.na.drop(subset=['Sales']).show()

df.na.fill('Fill Value').show()

df.na.fill(0).show()

df.na.fill('No Name', subset=['Name']).show()

from pyspark.sql.functions import mean

mean_val = df.select(mean(df['Sales'])).collect()

mean_sales = mean_val[0][0]

df.na.fill(mean_sales, subset = ['Sales']).show() # OR
df.na.fill(df.select(mean(df['Sales'])).collect()[0][0],['Sales']).show()
```

```
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('dates').getOrCreate()

df = spark.read.csv('appl_stock.csv', inferSchema=True, header=True)

df.select(['Date','Open']).show()

from pyspark.sql.functions import dayofmonth, hour, dayofyear, month, year, weekofyear, format_number, date_format

df.select(dayofmonth(df['Date'])).show()

df.select(year(df['Date'])).show()

df.withColumn('Year', year(df['Date'])).show()

newdf = df.withColumn('Year', year(df['Date']))

result = newdf.groupBy("Year").mean().select(['Year','avg(Close)'])

result.show()

new = result.withColumnRenamed('avg(Close)', 'Average Closing Price')

new.select(['Year', format_number('Average Closing Price', 2).alias('Avg Close')]).show()
```

```
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('lreg').getOrCreate()
from pyspark.ml.regression import LinearRegression

training = spark.read.format('libsvm').load('sample_linear_regression_data.txt')

training.show()

lr = LinearRegression(featuresCol = 'features', labelCol = 'label', predictionCol = 'prediction')

lrModel = lr.fit(training)

lrModel.coefficients

lrModel.intercept

training_summary = lrModel.summary

training_summary.r2

training_summary.rootMeanSquareError

all_data = spark.read.format('libsvm').load('sample_linear_regression_data.txt')

split_object = all_data.randomSplit([0.7, 0.3])

split_object

train_data, test_data = all_data.randomSplit([0.7, 0.3])

train_data.describe().show()

test_data.describe().show()

correct_model = lr.fit(train_data)

test_results = correct_model.evaluate(test_data)

test_results.residuals.show()

test_results.rootMeanSquareError

unlabeled_data = test_data.select('features')

unlabeled_data.show() 

predictions = correct_model.transform(unlabeled_data)

predictions.show()
```

```
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('lr_example').getOrCreate()

from pyspark.ml.regression import LinearRegression

data = spark.read.csv('Ecommerce_Customers.csv', inferSchema=True, header=True)

data.printSchema()

data.head(1)

for item in data.head(1)[0]:
	print(item)

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

data.columns 

assembler = VectorAssembler(inputCols = ['Avg Session Length', 'Time on App', 'Time on Website', 'Length of Memebership'],
			    outputCol = 'features')

output = assembler.transform(data)

output.select('features').show()

output.head(1)

final_data = output.select(['features', 'Yearly Amount Spent'])

final_data.show()

train_data, test_data = final_data.randomSplit([0.7, 0.3])

train_data.describe().show()

test_data.describe().show()

lr = LienarRegression(labelCol = 'Yearly Amount Spent')

lr_model = lr.fit(train_data)

test_results = lr_model.evaluate(test_data)

test_results.residuals.show()

test_results.rootMeanSquaredError

test_results.r2

final_data.describe().show()

unlabeled_data = test_data.select('features')

unlabeled_data.show()

predictions = lr_model.transform(unlabeled_data)

predictions.show()
```

```
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('cruise').getOrCreate()

df = spark.read.csv('cruise_ship_info.csv', inferSchema=True, header=True)

df.printSchema()

for ship in df.head(5):
	print(ship)
	print('\n')

df.groupBy('Cruise_line').count().show()

from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol = 'Crusie_line', outputCol = 'Cruise_cat')

indexed = indexer.fit(df).transform(df)

indexed.head(3)

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

indexed.columns

assembler = VectorAssembler(inputCols=['Age', 'Tonnage', 'passengers', 'length', 'cabins', 'passenger_density', 'crew', 'cruise_cat'],
			    outputCol=['features'])

output = assembler.transform(indexed)
output.select('features', 'crew').show()

final_data = output.select(['features', 'crew'])

train_data, test_data = final_data.randomSplit([0.7, 0.3])

train_data.describe().show()

test_data.describe().show()

from pyspark.ml.regression import LinearRegression

ship_lr = LinearRegression(labelCol = 'crew')

trained_ship_model = ship_lr.fit(train_data)

ship_results = trained_ship_model.evaluate(test_data)

ship_results.rootMeanSquaredError

train_data.describe().show()

ship_results.r2

ship_results.meanSquaredError

ship_results.meanAbsoluteError

from pyspark.sql.functions import corr

df.select(corr('crew', 'passengers')).show()

df.select(corr('crew', 'cabins')).show()  
```

```
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('mylogreg').getOrCreate()

from pyspark.ml.classification import LogisticRegression

my_data = spark.read.format('libsvm').load('sample_libsvm_data.txt')

my_data.show()

my_log_reg_model = LogisticRegression()

fitted_logreg = my_log_reg_model.fit(my_data)

log_summary = fitted_logreg.summary

log_summary.predictions.printSchema()

log_summary.predictions.show()

lr_train, lr_test = my_data.randomSplit([0.7, 0.3])

final_model = LogisticRegression()

fit_final = final_model.fit(lr_train)

predictions_and_labels = fit_final.evaluate(lr_test)

predictions_and_labels.predictions.show() 

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

my_eval = BinaryClassificationEvaluator()

my_final_roc = my_eval.evaluate(prediction_and_labels.predictions)

my_final_roc
```

```
(In databricks)

df = spark.sql("SELECT * FROM titanic_csv")

df.printSchema()

df.columns

my_cols = df.select(['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

my_final_data = my_cols.na.drop()

from pyspark.ml.feature import VectorAssembler, VectorIndexer, OneHotEncoder, StringIndexer

gender_indexer = StringIndexer(inputCol = 'Sex', outputCol = 'SexIndex')

gender_encoder = OneHotEncoder(inputCol = 'SexIndex', outputCol = 'SexVec')

embark_indexer = StringIndexer(inputCol = 'Embarked', outputCol = 'EmbarkIndex')

embark_encoder = OneHotEncoder(inputCol = 'EmbarkIndex', outputCol = 'EmbarkVec')

assembler = VectorAssembler(inputCols = ['Pclass', 'SexVec', 'EmbarkVec', 'Age', 'SibSp', 'Parch', 'Fare'],
			    outputCol = 'features')

from pyspark.ml.classification import LogisticRegression

from pyspark.ml import Pipeline

log_reg_titanic = LogisticRegression(featuresCol='features', labelCol='Survived')

pipeline = Pipeline(stages=[gender_indexer, embark_indexer, gender_encoder, embark_encoder, assembler, log_reg_titanic])

train_data, test_data = my_final_data.randomSplit([0.7, 0.3])

fit_model = pipeline.fit(train_data)

results = fit_model.transform(test_data)

from pyspark.ml.evaluation import BinaryClassificationEvaluator

my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='Survived')

results.select('Survived', 'prediction').show()

AUC = my_eval.evaluate(results)

AUC 
```

```
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('logregconsult').getOrCreate()

data = spark.read.csv('customer_churn.csv', inferSchema=True, header=True)

data.printSchema()

data.descibe().show()

data.columns

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols = ['Age', 'Total_Purchase', 'Account_Manager', 'Years', 'Num_Sites'],
			    outputCol = 'features')

output = assembler.transform(data)

final_data = output.select('features', 'churn')

train_churn,test_churn = final_data.randomSplit([0.7, 0.3])

from pyspark.ml.classification import LogisticRegression

lr_churn = LogisticRegression(labelCol='churn')

fitted_churn_model = lr_churn.fit(train_churn)

training_sum = fitted_churn_model.summary

training_sum.predictions.describe().show()

from pyspark.ml.evaluation import BinaryClassificationEvaluation

pred_and_labels = fitted_churn_model.evaluate(test_churn)

pred_and_labels.predictions.show()

churn_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='churn')

auc = churn.eval.evaluate(pred_and_labels.predictions)

auc

final_lr_model = lr_churn.fit(final_data)

new_customers = spark.read.csv('new_customers.csv', inferSchema=True, header=True)

new_customers.printSchema()

test_new_customers = assembler.transform(new_customers)

test_new_customers.printSchema()

final_results = final_lr_model.transform(test_new_customers)

final_results.select('Company','prediction').show()

test_new_customers.describe().show()
```

```
from pyspark.sql import SparkSession

spark = SaprkSession.builder.appName('mytree').getOrCreate()

from pyspark.ml import Pipeline

from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, DecisionTreeClassifier

data = spark.load.format('libsvm').load('sample_libsvm_data.txt')

data.show()

train_data, test_data = data.randomSplit([0.7, 0.3])

dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
gbt = GBTClassifier()

dtc_model = dtc.fit(train_data)
rfc_model = rfc.fit(train_data)
gbt_model = gbt.fit(train_data)

dtc_preds = dtc_model.transform(test_data)
rfc_preds = rfc_model.transform(test_data)
gbt_preds = gbt_model.transform(test_data)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

acc_eval = MulticlassClassificationEvaluator(metricName='accuracy')

print('DTC ACCURACY:')
acc_eval.evaluate(dtc_preds)

print('RFC ACCURACY:')
acc_eval.evaluate(rfc_preds)

print('GBT ACCURACY:')
acc_eval.evaluate(gbt_preds)

rfc_model.featureImportances
```

```
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('tree').getOrCreate()

data = spark.read.csv('College.csv', inferSchema=True, header=True)

data.printSchema()

data.head(1)

from pyspark.ml.feature import VectorAssembler

data.columns

assembler = VectorAssembler(inputCols=['Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F_Undergrad', 'P_Undergrad', 
			    'Outstate', 'Room_Board', 'Books', 'Personal', 'PhD', 'Terminal', 'S_F_Ratio', 'perc_alumni',
			    'Expend', 'Grad_Rate'], outputCol = 'features')

output = assembler.transform(data)

from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol='Private', outputCol='PrivateIndex')

output_fixed = indexer.fit(output).transform(output)

final_data = output_fixed.select('features', 'PrivateIndex')

train_data, test_data = final_data.randomSplit([0.7, 0.3])

from pyspark.ml.classification import DecisionTreeClassifier, GBTClassifier, RandomForestClassifier

from pyspark.ml import Pipeline

dtc = DecisionTreeClassifier(labelCol='PrivateIndex', featuresCol='features')
rfc = RandomForestClassifier(labelCol='PrivateIndex', featuresCol='features')
gbt = GBTClassifier(labelCol='PrivateIndex', featuresCol='features')

dtc_model = dtc.fit(train_data)
rfc_model = rfc.fit(train_data)
gbt_model = gbt.fit(train_data)

dtc_preds = dtc_model.transform(test_data)
rfc_preds = rfc_model.transform(test_data)
gbt_model = gbt_model.transform(test_data)

from pyspark.ml.evaluation import BinaryClassificationEvaluator

my_binary_eval = BinaryClassificationEvaluator(labelCol='PrivateIndex')

print('DTC')
print(my_binary_eval.evaluate(dtc_preds))

print('RFC')
print(my_binary_eval.evaluate(rfc_preds))

rfc_preds.printSchema()

gbt_preds.printSchema()

my_binary_eval2 = BinaryClassificationEvaluator(labelCol='PrivateIndex', rawPredictionCol='prediction')

print('GBT')
print(my_binary_eval2.evaluate(gbt_preds)) # Try changing default parameters to improve the score

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

acc_eval = MulticlassClassificationEvaluator(labelCol = 'PrivateIndex', metricName='accuracy')

rfc_acc = acc_eval.evaluate(rfc_preds)
```

```
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('tree_consult').getOrCreate()

data = spark.read.csv('dog_food.csv', inferSchema=True, header=True)

data.head(1)

from pyspark.ml.feature import VectorAssembler

data.columns

assembler = VectorAssembler(inputCols=['A', 'B', 'C', 'D'], outputCol= 'features')

output = assembler.transform(data)

from pyspark.ml.classification import RandomForestClassfier

rfc = RandomForestClassifier(labelCol='Spoiled', featuresCol='features')

output.printSchema()

final_data = output.select(['features', 'Spoiled'])

final_data.show()

rfc_model = rfc.fit(final_data)

final_data.head(1)

rfc_model.featureImportance
```

```
from pyspark.sql import SparkSession

spark = SaprkSession.builder.appName('cluster').getOrCreate()

from pyspark.ml.clustering import KMeans

dataset = spark.read.format('libsvm').load('sample_kmeans_data.txt')

dataset.show()

final_data = dataset.select('features')

final_data.show()

kmeans = KMeans().setK(2).setSeed(1)

model = kmeans.fit(final_data)

wssse = model.computeCost(final_data)

print(wssse)

centers = model.clusterCenters()

centers

results = model.transform(final_data)

results.show()
```

```
from spark.sql import SparkSession

spark = SparkSession.builder.appName('cluster').getOrCreate()

dataset = spark.read.csv('seeds_dataset.csv'. inferSchema=True, header=True)

dataset.printSchema()

dataset.head(1)

from pyspark.ml.clustering import KMeans

from pyspark.ml.feature import VectorAssembler

dataset.columns

assembler = VectorAssembler(inputCols=dataset.columns, outputCol='features')

final_data = assembler.transform(dataset)

final_data.printSchema()

from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures')

scaler_model = scaler.fit(final_data)

final_data = scaler_model.transform(final_data)

final_data.head(1)

kmeans = KMeans(featuresCol='scaledFeatures',k=3)

model = kmeans.fit(final_data)

print("WSSSE")
print(model.computeCost(final_data))

centers = model.clusterCenters()

print(centers)

model.transform(final_data).select('prediction').show()
```

```
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('cluster').getOrCreate()

dataset = spark.read.csv('hack_data.csv', inferSchema=True, header=True)

dataset.head()

from pyspark.ml.clustering import KMeans

from pyspark.ml.feature import VectorAssembler

dataset.columns

feat_cols = ['Session_Connection_Time', 'Bytes Transferred', 'Kali_Trace_Used', 'Servers_Corrupted', 'Pages_Corrupted', 	   	       'WPM_Typing_Speed']

assembler = VectorAssembler(inputCols=feat_cols, outputCol = 'features')

final_data = assembler.transform(dataset)

final_data.printSchema()

from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures')

scaler_model = scaler.fit(final_data)

cluster_final_data = scaler_model.transform(final_data)

cluster_final_data.printSchema()

kmeans2 = KMeans(featuresCol='scaledFeatures', k=2)
kmeans3 = KMeans(featuresCol='scaledFeatures', k=3)

model_k2 = kmeans2.fit(cluster_final_data)
model_k3 = kmeans3.fit(cluster_final_data)

model_k3.transform(cluster_final_data).select('prediction').show()

model_k3.transform(cluster_final_data).groupBy('prediction').count().show()

model_k2.transform(cluster_final_data).groupBy('prediction').count().show()
```

```
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('rec').getOrCreate()

from pyspark.ml.recommendation import ALS

from pyspark.ml.evaluation import RegressionEvaluator

data = spark.read.csv('movielens_ratings.csv', inferSchema=True, header=True)

data.show()

data.describe().show()

training, test = data.randomSplit([0.8, 0.2])

als = ALS(maxIter=5, regParam=0.01, userCol='userId',itemCol='movieId', ratingCol='rating')

model = als.fit(training)

predictions = model.transform(test)

predictions.show()

evaluator = RegressionEvaluator(metricName='rmse',labelCol='rating',predictionCol='prediction')

rmse = evaluator.evaluate(predictions)

print('RMSE')
print(rmse)

single_user = test.filter(test['userId']==11).select(['movieId', 'userId'])

single_user.show()

recommendations = model.transform(single_user)

recommendations.orderBy('prediction', ascending=False).show()
```

```
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('nlp').getOrCreate()

from pyspark.ml.feature import Tokenizer, RegexTokenizer

from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType

sen_df = spark.createDataFrame([
	(0, 'Hi I heard about Spark'),
	(1, 'I wish java could use case classes'),
	(2, 'Logistic,regression,models,are,neat')
], ['id', 'sentence'])

sen_df.show()

tokenizer = Tokenizer(inputCol='sentence', outputCol='words')

regex_tokenizer = RegexTokenizer(inputCol='sentence', outputCol='words', pattern='\\W')

count_tokens = udf(lambda words:len(words), IntegerType())

tokenized = tokenizer.transform(sen_df)

tokenized.show()

tokenized.withColumn('tokens', count_tokens(col('words'))).show()

rg_tokenized = regex_tokenizer.transform(sen_df)

rg_tokenized.show()

rg_tokenized.withColumn('tokens', count_tokens(col('words'))).show()

from pyspark.ml.feature import StopWordsRemover

sentenceDataFrame = spark.createDataFrame([
	(0,['I','saw','the','green','horse']),
	(1,['Mary','had','a','little','lamb'])],
	['id','tokens']
)

remover = StopWordsRemover(inputCol='tokens', outputCol='filtered')

remover.transform(sentenceDataFrame).show()

from pyspark.ml.feature import NGram

wordDataFrame = spark.createDataFrame([	
	(0, ['Hi', 'I', 'heard', 'about', 'Spark']),
	(1, ['I', 'wish', 'Java', 'could', 'use', 'case', 'classes']),
	(2, ['Logistic, 'regression', 'models', 'are', 'neat'])
], ['id', 'words'])

ngram = NGram(n=2, inputCol='words', outputCol='grams')

ngram.transform(wordDataFrame).select('grams').show(truncate=False)
```

```
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('nlp').getOrCreate()

from pyspark.ml.feature import HashingTF, IDF, Tokenizer

sentenceData = spark.createDataFrame([
	(0.0, 'Hi I heard about Spark'),
	(0.0, 'I wish Java could use case classes'),
	(1.0, 'Logistic regression models are neat')
], ['label', 'sentence'])

sentenceData.show()

tokenizer = Tokenizer(inputCol='sentence', outputCol='words')

words_data = tokenizer.transform(sentenceData)

words_data.show(truncate=False)

hashing_tf = HashingTF(inputCol='words', outputCol='rawFeatures')

featurized_data = hashing_tf.transform(words_data)

idf = IDF(inputCol='rawFeatures', outputCol='features')

idf_model = idf.fit(featurized_data)

rescaled_data = idf_mdoel.transform(featurized_data)

rescaled_data.select('label', 'features').show()

from pyspark.ml.feature import CountVectorizer

df = spark.createDataFrame([
	(0, "a b c".split(" ")),
	(1, "a b b c a".split(" "))
], ["id", "words"])

df.show()

cv = CountVectorizer(inputCol='words', outputCol='features', vocabSize=3, minDF=2.0)

model = cv.fit(df)

result = model.transform(df)

result.show(truncate=False) 
```

```
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('nlp').getOrCreate()

data = spark.read.csv('SMSSpamCollection', inferSchema=True, header=True, sep='\t')

data.show()

data = data.withColumnRenmaed('_c0', 'class').withColumnRenamed('_c1','text')

data.show()

from pyspark.sql.functions import length

data = data.withColumn('length', length(data['text']))

data.show()

data.groupBy('class').mean().show()

from pyspark.ml.feature import (Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer)

tokenizer = Tokenizer(inputCol='text', outputCol='token_text')
stop_remove = StopWordsRemover(inputCol='token_text', outputCol='stop_token')
count_vec = CountVectorizer(inputCol='stop_token', outputCol='c_vec')
idf = IDF(inputCol='c_vec', outputCol='tf_idf')
ham_spam_to_numeric = StringIndexer(inputCol='class', outputCol='label')

from pyspark.ml.feature import VectorAssembler

clean_up = VectorAssembler(inputCols=['tf_idf','length'], outputCol = 'features')

from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes()

from pyspark.ml import Pipeline

data_prep_pipe = Pipeline(stages=[ham_spam_to_numeric, tokenizer, stop_remove, count_vec, idf, clean_up])

cleaner = data_prep_pipe.fit(data)

clean_data = cleaner.transform(data)

clean_data = clean_data.select(['features','label'])

clean_data.show()

training, test = clean_data.randomSplit([0.7, 0.3])

spam_detector = nb.fit(training)

data.printSchema()

test_results = spam_detector.transform(test)

test_results.show()

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

acc_eval = MulticlassClassificationEvaluator()

acc = acc_eval.evaluate(test_results)

print('ACC of NB Model')
print(acc)
```

```
from pyspark import SparkContext

from pyspark.streaming import StreamingContext

sc = SparkContext('local[2]',NetworkWordCount)

ssc = StreamingContext(sc,1)

lines = ssc.socketTextStream('localhost',9999)

words = lines.flatMap(lambda line: line.split(' '))

pairs = words.map(lambda word:(word, 1))

word_counts = pairs.reduceByKey(lambda num1, num2: num1+num2)

words_counts.pprint()

ssc.start()
```

```
import tweepy
from tweepy import OAuthHandler, Stream

From tweepy.streaming import StreamListener
import socket
import json

consumer_key = ''
consumer_secret = ''
access_token = ''
access_secret = ''
```























