## SQL Bootcamp

### Notes

- Database: 1) A collection of data, 2) A method for accessing and manipulating that data
- Database vs Database Management System (DBMS)
- PostgreSQL, MySQL, Oracle Database, SQLite are database management systems
- MySQL vs SQL (Structured Query Language)
- SQL is the language we use to talk to our databases
-  



### SQL commands

- Example
```
SELECT * FROM customers; (returns all the data from customers table)

SELECT * 
FROM products 
ORDER BY Price DESC; (order by price descending)
```

- To start MySQL
```
```

- List current databases that exists in the MySQL server.
```
show databases;
```

- To create database
```
CREATE  DATABASE <name>;
```

- To delete a database
```
DROP DATABASE <name>;
```

- Using database
```
USE <database name>;
```

- To see currently used database 
```
SELECT database();
```

- Datatypes in SQL
Add a pic from Udemy

- Creating a table
```
CREATE TABLE tablename (
column_name data_type,
column_name data_type
);
```

- Show existing tables
```
SHOW TABLES;
```

- Show columns in a table
```
SHOW COLUMNS FROM <table_name>;

or

DESC <table_name>;
```

- Deleting tables
```
DROP TABLE <table_name>;
```

- Inserting into table
```
INSERT INTO cats(name, age)
VALUES ('Jetson', 7);
```

- To view data in the table
```
SELECT * FROM cats;
```

- Multiple insert into a table
```
INSERT INTO cats(name, age)
VALUES ('Charlie', 10)
      ,('Sadie', 3)
      ,('Lazy Bear', 1);
```

-  



























