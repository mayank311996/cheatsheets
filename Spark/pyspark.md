# PySpark

## Data LifeCycle

```
Collect --> Analyze --> Clean --> Organize --> Transform --> Insight
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

```











































