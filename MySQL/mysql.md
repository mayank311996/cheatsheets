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

- Work with parts of strings
```
SELECT SUBSTRING('Hello World', 1, 4);
SELECT SUBSTR('Hello World', 1, 4);
SELECT SUBSTRING('Hello World', 7);
SELECT SUBSTRING('Hello World', -3);
SELECT SUBSTRING(title, 1, 10) FROM books;
SELECT CONCAT(SUBSTRING(title, 1, 10), '...') AS 'short title' FROM books;
```

- To replace parts of strings
```
SELECT REPLACE('Hello World', 'Hell', '####');
SELECT REPLACE(title, 'e', '3') FROM books;
SELECT SUBSTRING(REPLACE(title, 'e', '3'), 1, 10) FROM books;
```
- To reverse
```
SELECT REVERSE('Hello World');
SELECT REVERSE(author_fname) FROM books;
SELECT CONCAT(author_fname, REVERSE(author_fname)) FROM books;
```

- To count characters in a string
```
SELECT CHAR_LENGTH('Hello World');
SELECT author_lname, CHAR_LENGTH(author_lname) AS 'length' FROM books;
SELECT CONCAT(author_lname, ' is ', CHAR_LENGTH(author_lanme), ' characters long') FROM books;
```

- You can also use SQL formatter

- To change a string's case
```
SELECT UPPER('Hello World');
SELECT LOWER('Hello World');
SELECT UPPER(title) FROM books;
SELECT CONCAT('My favourite book is the ', UPPER(title)) FROM books;
```

- To get distinct values
```
SELECT DISTINCT author_lname FROM books;
SELECT DISTINCT CONCAT(author_fname, ' ', author_lname) FROM books;
SELECT DISTINCT author_fname, author_lname FROM books;
```

- To sort results
```
SELECT author_lname FROM books ORDER BY author_lname; (ascending (ASC) by default)
SELECT author_lname FROM books ORDER BY author_lname DESC;
SELECT title, released_year, pages FROM books ORDER BY released_year;
SELECT title, author_fname, author_lname FROM books ORDER BY 2;
SELECT author_fname, author_lname FROM books ORDER BY author_lname, author_fname;
```

- To limit results 
```
SELECT title FROM books LIMIT 3;
SELECT title, released_year FROM books ORDER BY released_year DESC LIMIT 5;
SELECT title, released_year FROM books ORDER BY released_year DESC LIMIT 0,5; (first index and how many to go from there)
SELECT title FROM books LIMIT 5, 15461561651321651; (gigantic number to have from 5 to the end)
```

- To search better
```
SELECT title, author_fname FROM books WHERE author_fname LIKE '%da%';
SELECT title FROM books WHERE title LIKE '%the%';
SELECT title FROM books WHERE title LIKE '%\%the%';
SELECT title FROM books WHERE stock_quantity LIKE '____';
```

- The count function
```
SELECT COUNT(*) FROM books;
SELECT COUNT(author_lname) FROM books;
SELECT COUNT(DISTINCT author_lname) FROM books;
SELECT COUNT(DISTINCT author_lname, author_fname) FROM books;
SELECT COUNT(*) FROM books WHERE title LIKE '%the%';
```

- The group by
```

```










