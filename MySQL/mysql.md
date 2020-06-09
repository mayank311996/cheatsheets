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

- To see warnings 
```
SHOW WARNINGS;
```

- To define not null (default is null) columns
```
CREATE TABLE cats2 (
name VARCHAR(100) NOT NULL,
age INT NOT NULL);
```

- Setting default values
```
CREATE TABLE cats3 (
name VARCHAR(100) DEFAULT 'unnamed',
age INT DEFAULT 99
);
```

- To define not null and default values
```
CREATE TABLE cats3 (
name VARCHAR(100) NOT NULL DEFAULT 'unnamed',
age INT NOT NULL DEFAULT 99
);
```

- To define primary key (a unique identifier)
```
CREATE TABLE unique_cats (
cat_id INT NOT NULL,
name VARCHAR(100),
age INT,
PRIMARY KEY (cat_id));
```

- Auto incrementing primary key
```
CREATE TABLE unique_cats (
cat_id INT NOT NULL AUTO_INCREMENT,
name VARCHAR(100),
age INT,
PRIMARY KEY (cat_id));
```

- To view column in a table
```
SELECT name FROM cats;
SELECT name, age FROM cats;
```

- The WHERE clause, to get more specific
```
SELECT * FROM cats WHERE age=4;
```

- Aliases to modify read
```
SELECT cat_id AS id, name FROM cats;
```

- To update existing data
```
UPDATE cats SET breed='Shorthair'
WHERE breed ='Tabby';
```

- To delete things
```
DELETE FROM cats WHERE name='Egg';
DELETE FROM cats;
```

- To run sql file
```
SOURCE first_file.sql; 
```

- To combine data for cleanre output
```
SELECT CONCAT('Hello', 'World');
SELECT CONCAT(author_fname, ' ', author_lname) FROM books;
SELECT CONCAT(author_fname, ' ', author_lname) AS 'full name' FROM books;
SELECT author_fname AS first, author_lname AS last,
CONCAT(author_fname, ' ', author_lname) AS full
FROM books;
```

- To concat with separator
```
SELECT CONCAT_WS('-', title, author_name, author_lname) FROM books;
```

- 


















