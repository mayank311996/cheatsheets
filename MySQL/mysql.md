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
SELECT title, author_lname FROM books GROUP BY author_lname;
SELECT author_lname, COUNT(*) FROM books GROUP BY author_lname;
SELECT author_fname, author_lname, COUNT(*) FROM books GROUP BY author_lname;
SELECT author_fname, author_lname, COUNT(*) FROM books GROUP BY author_lname, author_fname;
SELECT released_year, COUNT(*) FROM books GROUP BY released_year;
SELECT CONCAT('In', released_yaer, COUNT(*), 'book(s) released') FROM books GROUP BY released_year;
```

- The min and max
```
SELECT MIN(released_year) FROM books;
SELECT MIN(pages) FROM books;
SELECT MAX(released_year) FROM books;
SELECT * FROM books 
WHERE pages = (SELECT Min(pages) FROM books); (query inside a query)
SELECT * FROM books ORDER BY pages ASC LIMIT 1;
```

- The min, max with group by
```
SELECT author_fname, 
       author_lname,
       MIN(released_year)
FROM   books
GROUP BY author_lname,
         author_fname;

SELECT author_fname, author_lname, MAX(pages) FROM books GROUP BY author_lname, author_fname;

SELECT 
CONCAT(author_fname, ' ', author_lname) AS author,
MAX(pages) AS 'longest book'
FROM books
GROUP BY author_lname, author_fname;
```
 
- The sum function
```
SELECT SUM(pages) FROM books;
SELECT author_fname, author_lname, SUM(pages)
FROM books
GROUP BY author_fname, author_lname;
```

- The avg function
```
SELECT AVG(released_year) FROM books;

SELECT AVG(stock_quantity) FROM books
GROUP BY released_year;

SELECT author_fname, author_lname, AVG(pages) FROM books
GROUP BY author_lname, author_fname;
```

- CHAR (fixed length)
```
CREATE TABLE dogs (name CHAR(5), breed VARCHAR(10));
```

- DECIMAL
```
CREATE TABLE items (price DECIMAL(5,2));
```

- FLOAT and DOUBLE
```
CREATE TABLE items (price FLOAT);
```

- DATE, TIME, DATETIME
```
CREATE TABLE people (
name VARCHAR(100),
birthdate DATE,
birthtime TIME,
birthdt DATETIME
);
```

- CURDATE, CURTIME, NOW
```
SELECT CURDATE();
SELECT CURTIME();
SELECT NOW();
```

- Formatting DATES
```
SELECT name, DAY(birthdate) FROM people;
SELECT name, DAYNAME(birthdate) FROM people;
SELECT name, DAYOFYEAR(birthdate) FROM people;
SELECT name, DAYOFWEEK(birthdate) FROM people;
SELECT name, MONTH(birthdate) FROM people;
SELECT name, MONTHNAME(birthdt) FROM people;
SEELCT name, HOUR(birthtime) FROM people;
SELECT name, MINUTE(birthtime) FROM people;
SELECT DATE_FORMAT('2009-10-04 22:23:00', '%W %M %Y');
```

- DATE math
```
SELECT DATEDIFF(NOW(), birthdate) FROM people;
SELECT birthdt, DATE_ADD(birthdt, INTERVAL 1 MONTH) FROM people;
SEELCT birthdt, birthdt + INTERVAL 1 MONTH FROM people;
```

- TIMESTAMPS
```
CREATE TABLE comments(
content VARCHAR(100),
created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE comments2(
content VARCHAR(100),
changed_at TIMESTAMP DEFAULT NOW() ON UPDATE CURRENT_TIMESTAMP
);
```

- Not Equal
```
SELECT title FROM books WHERE year != 2017;
```

- Not like
```
SELECT title FROM books WHERE title LIKE 'W%';
SELECT title FROM books WHERE title NOT LIKE 'W%';
```

- Greater than, Less than, Greater than or equal to, Less than or equal to
```
SELECT title, released_year FROM books WHERE released_year > 2000 ORDER BY released_year;
SELECT title, released_year FROM books WHERE released_year >= 2000 ORDER BY released_year;
SELECT title, released_year FROM books WHERE released_year < 2000 ORDER BY released_year;
SELECT title, released_year FROM books WHERE released_year <= 2000 ORDER BY released_year;
SELECT 99 > 1
```

- Logical AND(&&)
```
SELECT * FROM books 
WHERE author_lname='Eggers' 
AND released_year > 2010;

SELECT * FROM books 
WHERE author_lname='Eggers' AND
released_year > 2010 AND
title LIKE '%novel%';
```

- Logical OR(||)
```
SELECT * FROM books 
WHERE author_lname='Eggers' 
OR released_year > 2010;
```

- BETWEEN and NOT BETWEEN
```
SELECT title, released_year FROM 
books WHERE released_year >= 2004 AND
released_yaer <= 2015;

SELECT title, released_year FROM books
WHERE released_yaer BETWEEN 2004 AND 2015;

SELECT title, released_year FROM books
WHERE released_yaer NOT BETWEEN 2004 AND 2015;

SELECT name, birthdt FROM people 
WHERE birthdt BETWEEN CAST('1980-01-01' AS DATETIME)
AND CAST('2000-01-01' AS DATETIME);
```

- IN and NOT IN
```
SELECT title, author_lname FROM books
WHERE author_lname='Carver' OR
author_lname = 'Lahiri' OR
author_lname = 'Smith';

SELECT title, author_lname FROM books
WHERE author_lname IN ('Carver', 'Lahiri', 'Smith');

SELECT title, author_lname FROM books
WHERE author_lname NOT IN ('Carver', 'Lahiri', 'Smith');

SELECT title, released_year FROM books
WHERE released_year >= 2000
AND released_year NOT IN
(2000,2002,2004,2006,2008,2010,2012,2014,2016);
```

- MODULO(%)
```
SELECT title, released_year FROM books
WHERE released_year >= 2000
AND released_year % 2 != 0
ORDER BY released_year;
```

- CASE STATEMENTS
```
SELECT title, released_year,
	CASE
		WHEN released_year >= 2000 THEN 'Modern Lit'
		ELSE '20th century lit'
	END AS GENRE
FROM books;

SELECT title, stock_quantity,
	CASE
		WHEN stock_quantity BETWEEN 0 AND 50 THEN '*'
		WHEN stock_quantity BETWEEN 51 AND 100 THEN '**'
		ELSE '***'
	END AS STOCK
FROM books;

SELECT title, stock_quantity,
	CASE
		WHEN stock_quantity <= 50 THEN '*'
		WHEN stock_quantity <= 100 THEN '**'
		ELSE '***'
	END AS STOCK
FROM books;
```

- One to many relationship
- Primary key and Foreign key
```
CREATE TABLE customers(
	id INT AUTO_INCREMENT PRIMARY KEY,
	first_name VARCHAR(100),
	last_name VARCHAR(100),
	email VARCHAR(100)
);
CREATE TABLE orders(
	id INT AUTO_INCREMENT PRIMARY KEY,
	order_date DATE,
	amount DECIMAL(8,2),
	customer_id INT,
	FOREIGN KEY(customer_id) REFERENCES customers(id)
);
```

- Cross JOIN
```
SELECT * FROM orders WHERE customer_id = 
	(
		SELECT id FROM customers
		WHERE last_name = 'George'
	);

SELECT * FROM customers, orders; (cross join)
```

- Inner Join
```
SELECT * FROM customers, orders
WHERE customers.id = orders.customer_id; (implicit inner join)

SELECT * FROM customers
JOIN orders 
ON customers.id = orders.customer_id; (explicit inner join) (order matters)

SELECT first_name, last_name, order_date, amount
FROM customers
JOIN orders
ON customers.id = orders.customer_id; (you can explicitly write INNER JOIN)
```

```
SELECT first_name, last_name, order_date, SUM(amount) AS total_spent
FROM customers
JOIN orders
	ON customers.id = orders.customer_id
GROUP BY orders.customer_id
ORDER BY total_spent;
```

- Left Join
```
SELECT * FROM customers
LEFT JOIN orders
	ON customers.id = orders.customer_id;

SELECT first_name, last_name, order_date, amount
FROM customers
LEFT JOIN orders
	ON cutomers.id = orders.customer_id;

SELECT first_name, last_name, SUM(amount) FROM cutomers 
LEFT JOIN orders
	ON customers.id = orders.customer_id
GROUP BY customers.id; 

SELECT first_name, last_name, IFNULL(SUM(amount, 0)) AS total_spent
FROM customers
LEFT JOIN orders
	ON customers.id = orders.customer_id
GROUP BY customers.id;
```

- Right Join
```
SELECT * FROM customers
RIGHT JOIN orders
	ON customers.id = orders.customer_id; 

SELECT 
	IFNULL(first_name, 'MISSING') AS first,
	IFNULL(last_name, 'USER') AS last,
	order_date,
	amount,
	SUM(amount),
FROM customers
RIGHT JOIN orders
	ON customers.id = orders.customer_id
GROUP BY customer_id;

CREATE TABLE orders(
	id INT AUTO_INCREMENT PRIMARY_KEY,	
	order_date DATE,
	amount DECIMAL(8,2),
	customer_id INT,
	FOREIGN KEY(customer_id)
		REFERENCES customes(id)
		ON DELETE CASCADE
);
```

- Many to Many
```
CREATE TABLE reviewers(
	id INT AUTO_INCREMENT PRIMARY KEY,
	first_name VARCHAR(100),
	last_name VARCHAR(100)
);
CRAETE TABLE series(
	id INT AUTO_INCREMENT PRIMARY KEY,
	title VARCHAR(100),
	released_year YEAR(4),
	genre VARCHAR(100) 
);
CREATE TABLE reviews(
	id INT AUTO_INCREMENT PRIMARY KEY,
	rating DECIMAL(2,1),
	series_id INT,
	reviewrs_id INT,
	FOREIGN KEY(series_id) REFERENCES series(id),
	FOREIGN KEY(reviewer_id) REFERENCES reviewers(id)
);

SELECT title, rating
FROM series
JOIN reviews
	ON series.id = reviews.series_id;

SELECT series.id, title, AVG(rating) AS avg_rating
FROM series 
JOIN reviews
	ON series.id = reviews.series_id
GROUP BY series.id;
ORDER BY avg_rating;

SELECT first_name, last_name, rating
FROM reviewers
JOIN reviews 
	ON reviewers.id = reviews.reviewer_id;

SELECT title AS unreviewed_series
FROM series
LEFT JOIN reviews
	ON series.id = reviews.series_id
WHERE rating IS NULL;

SELECT genre, ROUND(AVG(rating),2) AS avg_rating
FROM series
JOIN reviews
	ON series.id = reviews.series_id
GROUP BY genre;

SELECT
	first_name,
	last_name,
	COUNT(rating) AS COUNT,
	IFNULL(MIN(rating), 0) AS MIN,
	IFNULL(MAX(rating), 0) AS MAX,
	IFNULL(AVG(rating), 0) AS AVG,
	CASE
		WHEN COUNT(rating) >= 1 THEN 'ACTIVE'
		ELSE 'INACTIVE'
	END AS STATUS
FROM reviewers 
LEFT JOIN reviews
	ON reviewers.id = reviews.reviewer_id
GROUP BY reviewers.id;

SELECT
	first_name,
	last_name,
	COUNT(rating) AS COUNT,
	IFNULL(MIN(rating), 0) AS MIN,
	IFNULL(MAX(rating), 0) AS MAX,
	IFNULL(AVG(rating), 0) AS AVG,
	IF(COUNT(rating) >= 1, 'ACTIVE', 'INACTIVE') AS STATUS 
FROM reviewers 
LEFT JOIN reviews
	ON reviewers.id = reviews.reviewer_id
GROUP BY reviewers.id;

SELECT 
	title,
	rating,
	CONCAT(first_name, ' ', last_name) AS reviewer
FROM reviewers
JOIN reviews
	ON reviewers.id = reviews.reviewer_id
JOIN series
	ON series.id = reviews.series_id
ORDER BY title;
```
 
- Instagram schema
```
DROP DATABASE ig_clone;
CREATE DATABASE ig_clone;
USE ig_clone;

CREATE TABLE users(
	id INT AUTO_INCREMENT PRIMARY KEY,
	username VARCHAR(255) UNIQUE NOT NULL,
	created_at TIMESTAMP DEFAULT NOW()
); 

CREATE TABLE photos(
	id INT AUTO_INCREMENT PRIMARY KEY,
	image_url VARCHAR(255) NOT NULL,
	user_id INTEGER NOT NULL,
	created_at TIMESTAMP DEFAULT NOW(),
	FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE TABLE comments(
	id INT AUTO_INCREMENT PRIMARY KEY,
	comment_text VARCHAR(255 )    NOT NULL,
	user_id INT NOT NULL,
	photo_id INT NOT NULL,
	created_at TIMESTAMP DEFAULT NOW(),
	FOREIGN KEY(user_id) REFERENCES users(id),
	FOREIGN KEY(photo_id) REFERENCES photos(id)
);

CREATE TABLE likes(
	user_id INT NOT NULL,
	photo_id INT NOT NULL,
	created_at TIMESTAMP DEFAULT NOW(),
	FOREIGN KEY(user_id) REFERENCES users(id),
	FOREIGN KEY(photo_id) REFERENCES photos(id),
	PRIMARY KEY(user_id, photo_id) (just to confirm not more than one likes on same photo by same user)
); (no id here because we will be not referring that)

CREATE TABLE follows(
	follower_id INT NOT NULL,
	followee_id INT NOT NULL,
	created_at TIMESTAMP DEFAULT NOW(),
	FOREIGN KEY(follower_id) REFERENCES users(id),
	FOREIGN KEY(followee_id) REFERENCES users(id),
	PRIMARY KEY(follower_id, followee_id)
);

CREATE TABLE tags(
	id INT AUTO_INCREMENT PRIMARY KEY,
	tag_name VARCHAR(255) UNIQUE,
	created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE photo_tags(
	photo_id INT NOT NULL,
	tag_id INT NOT NULL,
	FOREIGN KEY(photo_id) REFERENCES photos(id),
	FOREIGN KEY(tag_id) REFERENCES tags(id),
	PRIMARY KEY(photo_id, tag_id)
);
```

- Working with Instagram schema
```
source starter_data.sql

SELECT COUNT(*) FROM likes;

SELECT * FROM users ORDER BY created_at LIMIT 5;

SELECT username, DAYNAME(created_at) FROM users;
SELECT DAYNAME(created_at) AS day,
COUNT(*) AS total
FROM users 
GROUP BY day
ORDER BY total DESC;

SELECT username 
FROM users
LEFT JOIN photos
	ON users.id = photos.user_id
WHERE photos.image_url IS NULL;

SELECT
	photos.id,
	photos.image_url,
	COUNT(*) AS total
FROM photos
INNER JOIN likes
	ON likes.photo_id = photos.id
GROUP BY photos.id
ORDER BY total DESC
LIMIT 1;
SELECT
	username,
	photos.id,
	photos.image_url,
	COUNT(*) AS total
FROM photos
INNER JOIN likes
	ON likes.photo_id = photos.id
INNER JOIN users
	ON photos.user_id = users.id
GROUP BY photos.id
ORDER BY total DESC
LIMIT 1;

```



























































