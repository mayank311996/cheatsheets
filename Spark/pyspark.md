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
 
```

















































