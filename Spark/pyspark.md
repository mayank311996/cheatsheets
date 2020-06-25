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

```

























