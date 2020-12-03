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
console.log(removedFromMyArray)

// ############# Manipulate arrays with shift

// Example
var ourArray = [1, 2, 3];
var removedFromOurArray = ourArray.shift();

// Setup
var myArray = [["John", 23], ["cat", 2]];

var removedFromMyArray = myArray.shift();
console.log(removedFromMyArray)

// ############# Manipulate arrays with unshift

// Example
var ourArray = ["Stimpson", "J", "cat"];
ourArray.shift();
ourArray.unshift("Happy");

// Setup
var myArray = [["John", 23], ["dog", 3]];
myArray.shift();
myArray.unshift(["kite", 1]);

// ############# Shopping list

var myList = [["cereal", 3], ["milk", 2], ["bananas", 3]]

// ############ Write reusable code with functions

// Example
function ourReusableFunction() {
    console.log("Heyya", "World");
}

ourReusableFunction();

// Only change code below this line
function reusableFunction() {
    console.log("Hi World");
}

reusableFunction();

// ############ Passing values to functions with arguments

// Example
function ourFunctionWithArgs(a, b) {
    console.log(a - b);
}
ourFunctionWithArgs(10, 5);

// Only change code below this line
function functionWithArgs(a, b) {
    console.log(a + b);
}
functionWithArgs(10, 5);

// ############# Global scope and functions

// Declare your variable here
var myGlobal = 10; // As this variable is defined outside, it has global scope
// and can be accessed inside functions too

function fun1() {
    // Assign 5 to oopsGlobal Here
    oopsGlobal = 5; // Here we didn't put var before oopsGlobal which makes it
    // global and can be accessed outside function and other functions too. If
    // we put var before oopsGlobal then scope of that is limited to present
    // function means we cannot access it outside this function
}

// Only change code above this line
function fun2() {
    var output = "";
    if (typeof myGlobal != "undefined") {
        output += "myGlobal: " + myGlobal;
    }
    if (typeof oopsGlobal != "undefined") {
        output += "oopsGlobal: " + oopsGlobal;
    }
    console.log(output);
}
fun1();
fun2();

// ############## Local scope and functions

function myLocalScope() {
    var myVar = 5;
    console.log(myVar); // local scope
}
myLocalScope();
console.log(myVar); // Will give an error

// ############## Global vs local scope in functions

// It is possible to have same names for both local and global variables.
// However, local variable will take over global variable.

var outerWear = "T-Shirt";

function myOutfit() {
    var outerWear = "sweater";

    return outerWear;
}

console.log(myOutfit());
console.log(outerWear);

// ############## Return a value from a function with return

function minusSeven(num) {
    return num - 7;
}
console.log(minusSeven(10));

function timesFive(num) {
    return num * 5;
}
console.log(timesFive(5));

// ############## Understanding undefined value returned from a function

// Example
var sum = 0;
function addThree() {
    sum = sum + 3;
}

function addFive() {
    sum += 5;
}

// ############## Assignment with a returned value

var changed = 0;

function change(num) {
    return (num + 5) / 3;
}
changed = change(10);

var processed = 0;
function processArg(num) {
    return (num + 3) / 5;
}
processed = processArg(7);

// ############### Stand in line

function nextInLine(arr, item) {
    // Your code here
    arr.push(item);
    return arr.shift();
}

var testArr = [1, 2, 3, 4, 5];

console.log("Before: " + JSON.stringify(testArr));
console.log(nextInLine(testArr, 6));
console.log("After: " + JSON.stringify(testArr));

// ################ Boolean values

function welcomeToBooleans() {
    return true;
}

// ################ Use conditional logic with if statements

// Example
function ourTrueOrFalse(isItTrue) {
    if (isItTrue) {
        return "Yes, it's true";
    }
    return "No, it's false";
}

function trueOrFalse(wasThatTrue) {
    if (wasThatTrue) {
        return "Yes, that was true";
    }
    return "No, that was false";
}
console.log(trueOrFalse(true));
console.log(trueOrFalse(false));

// ############### Comparison with the equality operator

// Setup
function testEqual(val) {
    if (val == 12) {
        return "Equal";
    }
    return "Not Equal";
}
console.log(testEqual(10));

// ############### Comparison with the strict equality operator

// Equality operator tries to convert both variables to same type before
// comparing while strict equality operator doesn't do any type conversion

// Setup
function testStrict(val) {
    if (val === 10) {
        return "Equal";
    }
    return "Not Equal";
}

testStrict(10);

// ############## Practice comparing different values

// Setup
function compareEquality(a, b) {
    if (a == b) {
        return "Equal";
    }
    return "Not Equal";
}
console.log(compareEquality(10, "10"));

// ############ Comparison with inequality operator

// Setup
function testNotEqual(val) {
    if (val != 99) {
        return "Not Equal";
    }
    return "Equal";
}
console.log(testNotEqual(10));

// ########### Comparison with the Strict Inequality Operator

// Setup
function testStrictNotEqual(val) {
    if (val !== 17) {
        return "Not Equal";
    }
    return "Equal";
}

console.log(testStrictNotEqual(10));

// ########### Comparison with the greater than operator

function testGreaterThan(val) {
    if (val > 100) {
        return "Over 100";
    }
    if (val > 10) {
        return "Over 10";
    }
    return "10 or under";
}
console.log(testGreaterThan(10));

// ########### Comparison with the greater than or equal to operator

function testGreaterThanOrEqual(val) {
    if (val >= 100) {
        return "100 or over";
    }
    if (val >= 10) {
        return "10 or over";
    }
    return "less than 10";
}
console.log(testGreaterThanOrEqual(10));

// ############ Comparison with the less than operator

// ############ Comparison with the less than or equal to operator

// ############ Comparison with the logical and operator

function testLogicalAnd(val) {
    if (val <= 50 && val >= 25) {
        return "Yes";
    }
    return "No";
}
testLogicalAnd(10);

// ########### Comparison with the logical or operator
// if (val < 10 || val > 20)

// ########### Else statement

function testElse(val) {
    var result = "";

    if (val > 5) {
        result = "Bigger than 5";
    } else {
        result = "5 or Smaller";
    }
    return result;
}
testElse(4);

// ############ Elseif statement
// order is really important in elseif statement

function testElseIf(val) {
    if (val > 10) {
        return "Greater than 10";
    } else if (val < 5) {
        return "Smaller than 5";
    } else {
        return "Between 5 and 10";
    }
}
testElseIf(7);

// ############ Chaining if else statements

function testSize(num) {
    if (num < 5) {
        return "Tiny";
    } else if (num < 10) {
        return "Small";
    } else if (num < 15) {
        return "Medium";
    } else if (num < 20) {
        return "Large";
    } else {
        return "Huge";
    }
}
console.log(testSize(7));

// ########### Golf code

var names = ["Hole-in-one!", "Eagle", "Birdie", "Par", "Bogey", "Double Bogey",
             "Go Home!"]

function golfScore(par, strokes) {
    if (strokes == 1) {
        return names[0];
    } else if (strokes <= par - 2) {
        return names[1];
    } else if (strokes == par - 1) {
        return names[2];
    } else if (strokes == par) {
        return names[3];
    } else if (strokes == par + 1) {
        return names[4];
    } else if (strokes == par + 2) {
        return names[5];
    } else if (strokes >= par + 3) {
        return names[6];
    }
    return "Change Me";
}
console.log(golfScore(5, 4));

// ############ Switch statements
// switch uses strict equality operator inside for comparison

function caseInSwitch(val) {
    var answer = "";
    switch (val) {
        case 1:
            answer = "alpha";
            break;
        case 2:
            answer = "beta";
            break;
        case 3:
            answer = "gamma";
            break;
        case 4:
            answer = "delta";
            break;
    }
    return answer;
}
console.log(caseInSwitch(2));

// ############# Default option in switch statements

function caseInSwitch(val) {
    var answer = "";
    switch (val) {
        case "a":
            answer = "alpha";
            break;
        case "b":
            answer = "beta";
            break;
        case "c":
            answer = "gamma";
            break;
        case "d":
            answer = "delta";
            break;
        default:
            answer = "stuff";
            break;
    }
    return answer;
}
console.log(caseInSwitch("b"));

// ############### Multiple identical options in switch statements

function caseInSwitch(val) {
    var answer = "";
    switch (val) {
        case "a":
        case "b":
        case "c":
            answer = "gamma";
            break;
        case "d":
        case "e":
        case "f":
            answer = "delta";
            break;
        default:
            answer = "stuff";
            break;
    }
    return answer;
}
console.log(caseInSwitch("b"));

// ################## Replacing if else chains with switch

// ################## Returning boolean values from functions

function isLess(a, b) {
    return a < b;
}
console.log(isLess(10, 15));

// ################## Returning early pattern from functions

function abTest(a, b) {
    if (a < 0 || b < 0) {
        return undefined;
    }
    return Math.round(Math.pow(Math.sqrt(a) + Math.sqrt(b), 2));
}
console.log(abTest(2, 2));

// ################### Counting cards

var count = 0;

function cc(card) {
    switch (card) {
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
            count++;
            break;
        case 10;
        case "J";
        case "Q";
        case "K";
        case "A";
            count--;
            break;
    }

    var holdbet = "Hold";
    if (count > 0) {
        holdbet = "Bet"
    }

    return count + " " + holdbet;
}
console.log(cc(3));

// #################### Build JavaScript objects

var ourDog = {
    "name": "Camper",
    "legs": 4,
    "tails": 1,
    "friends": ["everything"]
};

// #################### Accessing object properties with dot notation

var dogName = ourDog.name;
var dogLegs = ourDog.legs;

// #################### Accessing object properties with bracket notation
// This becomes necessary when variables have space in names - e.g. "name 12"

var dogName = ourDog["name"];
var dogLegs = ourDog["legs"];

// #################### Accessing object properties with variables

var testobj = {
    12: "Namath";
    16: "Montana";
    19: "unitas"
};

var playerNumber = 16;
var player = testobj[playerNumber];

// ################# Updating object properties

var ourDog = {
    "name": "Camper",
    "legs": 4,
    "tails": 1,
    "friends": ["everything"]
};

ourDog.name = "Happy Camper";

// ################ Add new properties to an object

var ourDog = {
    "name": "Camper",
    "legs": 4,
    "tails": 1,
    "friends": ["everything"]
};

ourDog.bark = "bow-wow";

// ################ Delete properties from an object

var ourDog = {
    "name": "Camper",
    "legs": 4,
    "tails": 1,
    "friends": ["everything"],
    "bark": "bow-wow"
};

delete ourDog.bark;

// ################ Using objects for lookups
// basically replacing switch statements with objects

// ################ Testing objects for properties

var myObj = {
    gift: "pony";
    pet: "kitten";
    bed: "sleigh"
};

function checkObj(checkProp) {
    if (myObj.hasOwnProperty(checkProp)) {
        return myObj[checkProp];
    } else {
        return "Not Found"
    }
}
console.log(checkObj("gift"));

// ################ Manipulating complex objects

var myMusic = [
    {
        "artist": "Billy Joel",
        "title": "Piano Man",
        "release_year": 1973,
        "formats": [
            "CD",
            "8T",
            "LP"
        ],
        "gold": true
    },
    {
        "artist": "Beau Carnes",
        "title": "Cereal Man",
        "release_year": 2001,
        "formats": [
            "CD",
            "8T",
            "LP"
        ],
        "gold": true
    }
]

// ################ Accessing nested objects

var myStorage = {
    "car": {
        "inside": {
            "glove box": "maps",
            "passenger seat": "crumbs"
        },
        "outside": {
            "trunk": "jack"
        }
    }
};

var gloveBoxContents = myStorage.car.inside["glove box"];
console.log(gloveBoxContents)

// ################# Accessing nested arrays

var myPlants = [
    {
        type: "flowers",
        list: [
            "rose",
            "tulip",
            "dandelion"
        ]
    },
    {
        type: "trees",
        list: [
            "fir",
            "pine",
            "birch"
        ]
    }
];

var secondTree = myPlants[1].list[1];

// ################# Record collection

var collection = {
    "2548": {
        "album": "Slippery When Wet",
        "artist": "Bon Jovi",
        "tracks": [
            "let It Rock",
            "You Give Love a Bad Name"
        ]
    },
    "2468": {
        "album": "1999",
        "artist": "Prince",
        "tracks": [
            "1999",
            "Little Red Corvette"
        ]
    },
    "1245": {
        "artist": "Robert Palmer",
        "tracks": []
    },
    "5439": {
        "album": "ABBA Gold"
    }
};

var collectionCopy = JSON.parse(JSON.stringify(collection));

function updateRecords(id, prop, value) {
    if (value === "") {
        delete collection[id][prop];
    } else if (prop === "tracks") {
        collection[id][prop] = collection[id][prop] || [];
        collection[id][prop].push(value);
    } else {
        collection[id][prop] = value;
    }
    return collection;
}
console.log(updateRecords(5439, "artist", "ABBA"));
console.log(updateRecords(2468, "tracks", "test"));

// ################# Iterate with while loops

var myArray = [];

var i = 0;
while(i < 5) {
    myArray.push(i);
    i++;
}
console.log(myArray);

// ################# Iterate with for loops

var ourArray = [];

for (var i = 0; i < 5; i++) {
    ourArray.push(i);
}
console.log(myArray);

// ################# Iterate Odd Numbers with a for loop

var ourArray = [];

for (var i = 0; i < 5; i+=2) {
    ourArray.push(i);
}
console.log(myArray);

// ################# Count backwards with a for loop

var ourArray = [];

for (var i = 10; i > 0; i -= 2) {
    ourArray.push(i);
}
console.log(myArray);

// ################ Iterate through an array with a for loop

var ourArray = [9, 10, 11, 12];
var ourTotal = 0;

for (var i = 0; i < ourArray.length; i++) {
    ourTotal += ourArray[i];
}
console.log(ourTotal);

// ############### Nesting for loops

function multiplyAll(arr) {
    var product = 1;

    for (var i = 0; i < arr.length; i++) {
        for (var j = 0; j < arr[i].length; j++) {
            product *= arr[i][j];
        }
    }
    return product;
}

var product = multiplyAll([[1, 2], [3, 4], [5, 6, 7]]);

// ################ Iterate with do...while loops
// A while loop checks condition first before executing even a single time
// whereas do...while loop executes for at least one time doesn't matter if
// condition is met or not

var myArray = [];
var i = 10;

do {
    myArray.push(i);
    i++;
} while (i < 5)
console.log(i, myArray);

// ################# Profile lookup

var contacts = [
    {
        "firstName": "Akira",
        "lastName": "Laine",
        "number": "0568767687",
        "likes": ["Pizza", "Coding", "Brownie Points"]
    },
    {
        "firstName": "Harry",
        "lastName": "Potter",
        "number": "0879036784",
        "likes": ["Hogwarts", "Magic", "Hagrid"]
    },
    {
        "firstName": "Sherlock",
        "lastName": "Holmes",
        "number": "0484694068",
        "likes": ["Intriguing Cases", "Violin"]
    },
    {
        "firstName": "Kristian",
        "lastName": "Vos",
        "number": "unknown",
        "likes": ["JavaScript", "Gaming", "Foxes"]
    },
]

function lookUpProfile(name, prop){
    for (var i = 0; i < contacts.length; i++) {
        if (contacts[i].firstName === name) {
            return contacts[i][prop] || "No such property"; // This is a fancy
            // way to avoid one more for loop. Basically, it checks for value
            // and if it cannot find value then return whatever is after "||"
        }
    }
    return "No such contact";
}

var data = lookUpProfile("Sherlock", "lastName");
console.log(data);

// ################### Generate random fractions

function randomFraction() {
    return Math.random();
}

console.log(randomFraction());

// ################### Generate random whole numbers

var randomNumberBetween0and19 = Math.floor(Math.random() * 20);

function randomWholeNum() {
    return Math.floor(Math.random() * 10);
}
console.log(randomWholeNum());

// ################### Generate random whole numbers within a range

function ourRandomRange(ourMin, ourMax) {
    return Math.floor(Math.random() * (ourMax - ourMin + 1)) + ourMin;
}
console.log(ourRandomRange(1, 9));

// ################### Use the parseInt function
// If it cannot convert string to int then returns NaN

function convertToInteger(str) {
    return parseInt(str);
}
convertToInteger("56");

// ################### Use the parseInt function with a Radix

function convertToInteger(str) {
    return parseInt(str, 2); // 2 after comma to indicate that it's a binary
    // number (string)
}
convertToInteger("10011");

// ################### Use the conditional (ternary) operator
// condition ? statement-if-true : statement-if-false;

function checkEqual(a, b) {
    return a === b ? true : false; // by default it returns true and false only
    // but we just used ternary operator for explanation
}
checkEqual(1, 2);

// #################### Use multiple conditional (ternary) operators

function checkSign(num) {
    return num > 0 ? "positive": num < 0 ? "negative" : "zero"
}
console.log(checkSign(0));

// ##################### Differences between the var and let keywords
// You can use var for same variables two times but not for let. It will
// give an error

let catName = "Quincy";
let catName = "car"; // gives an error

var catName = "quincy";
var catName = "car"; // works fine

let catName = "quincy";
catName = "car"; // works fine because we are just changing the value not
// defining variable again

function catTalk() {
    "use strict"; // catches bad practices for example if you declared a
    // variable and forgot to assign a value. This can be placed at top of the
    // file as well
    catName = "Oliver";
    quote = catName + "says Meow!";
}
catTalk();

// ###################### Compare scopes of the var and let keywords







































