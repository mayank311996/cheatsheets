// ############### Find length of a string

// Example
var firstNameLength = 0;
var firstName = "Ada";

firstNameLength = firstName.length;

// Setup
var lastNameLength = 0;
var lastName = "Lovelace";

// Only change code below this line

lastNameLength = lastName.length;
console.log(lastNameLength)

// ############### Bracket notation to find first character in string

// Example
var firstLetterOfFirstName = "";
car firstName = "Ada";

firstLetterOfFirstName = firstName[0]

// Setup
var firstLetterOfLastName = "";
var lastName = "Lovelace";

// Only change code below this line
firstLetterOfLastName = lastName[0];
console.log(firstLetterOfLastName)

// ################ String immutability

// Setup
var myStr = "Jello World";

// Only change code below this line
myStr[0] = "H"; // Will give error
myStr = "Hello World"; // Will run

// ############### Bracket notation to find Nth character in string

// Example
var firstName = "Ada";
var secondLetterOfFirstName = firstName[1];

// Setup
var lastName = "Lovelace";

// Only change code below this line
var thirdLetterOfLastName = lastName[2];

// ############### Bracket notation to find last letter in string

// Example
var firstName = "Ada";
var lastLetterOfFirstName = firstName[firstName.length - 1];

// Setup
var lastName = "Lovelace";

// Only change code below this line
var lastLetterOfLastName = lastName[lastName.length - 1];

// ############## Bracket notation to find Nth-to-Last character in string

// Example
var firstName = "Ada";
var thirdToLastLetterOfFirstName = firstName[firstName.length - 3];

// Setup
var lastName = "Lovelace";

// Only change code below this line
var secondToLastLetterOfLastName = lastName[lastName.length - 2];

// ############# Word blanks

function wordBlanks(myNoun, myAdjective, myVerb, myAdverb) {
    // Your code below this line
    var result = "";
    result += "The " + myAdjective + " " + myNoun + " " + myVerb +
    " to the store " + myAdverb
    // Your code above this line
    return result;
}

// Change the words here to test your function
console.log(wordBlanks("dog", "big", "ran", "quickly"))

// ################ Store multiple values with arrays

// Example
var ourArray = ["John", 23];

// Only change code below this line
var myArray = ["Quincy", 1];

// ############### Nested arrays

// Example
var ourArray = [["the universe", 42], ["everything", 101010]];

// Only change code below this line
var myArray = [["Bulls", 23], ["White", 20]];

// ############### Access array data with indexes

// Example
var ourArray = [50, 60, 70];
var ourData = ourArray[0];

// Setup
var myArray = [50, 60, 70];

// Only change code below this line
var myData = myArray[0];
console.log(myData)

// ################ Modify array data with indexes

// Example
var ourArray = [18, 64, 99];
ourArray[1] = 45;

// Setup
var myArray = [18, 64, 99];

// Only change code below this line
myArray[2] = 0;

// ############### Access multi-dimensional arrays with indexes

// Setup
var myArray = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [[10, 11, 12], 13, 14]];

// Only change code below this line
var myData = myArray[0][0];
console.log(myData)

// ############## Manipulate arrays with push

// Example
var ourArray = ["Stimpson", "J", "cat"];
ourArray.push(["happy", "joy"]);

//Setup
var myArray = [["John", 23], ["cat", 2]];

// Only change code below this line
myArray.push(["kite", 1]);

// ############# Manipulate arrays with pop

// Example
var ourArray = [1, 2, 3];
var removedFromOurArray = ourArray.pop();

// Setup
var myArray = [["John", 23], ["cat", 2]];

var removedFromMyArray = myArray.pop();

















































