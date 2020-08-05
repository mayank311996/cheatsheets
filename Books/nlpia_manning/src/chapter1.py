######################################################################################
# Chapter 1
######################################################################################

import re
from collections import Counter
from itertools import permutations
import numpy as np

#######################################################################################
r = "(hi|hello|hey)[ ]*([a-z]*)"
re.match(r, 'Hello Rosa', flags=re.IGNORECASE)
re.match(r, "hi ho, hi ho, it's off to work ...", flags=re.IGNORECASE)
re.match(r, "hey, what's up", flags=re.IGNORECASE)

r = r"[^a-z]*([y]o|[h']?ello|ok|hey|(good[ ])?(morn[gin']{0,3}|" \
    r"afternoon|even[gin']{0,3}))[\s,;:]{1,3}([a-z]{1,20})"
re_greeting = re.compile(r, flags=re.IGNORECASE)
re_greeting.match("Hello Rosa")
re_greeting.match("Hello Rosa").groups()
re_greeting.match("Good morning Rosa")
re_greeting.match("Good Manning Rosa")
re_greeting.match("Good evening Rosa Parks").groups()
re_greeting.match("Good Morn'n Rosa")
re_greeting.match("yo Rosa")

my_names = {"rosa", "rose", "chatty", "chatbot", "bot", "chatterbot"}  # Another way: set([])
curt_names = {"hal", "you", "u"}
greeter_name = ""
match = re_greeting.match(input())

if match:
    at_name = match.groups()[-1]
    if at_name in curt_names:
        print("Good one.")
    elif at_name.lower() in my_names:
        print(f"Hi {greeter_name}, how are you?")


#########################################################################################
Counter("Guten Morgen Rosa".split())
Counter("Good morning, Rosa!".split())


#########################################################################################
[" ".join(combo) for combo in permutations("Good morning "
                                           "Rosa!".split(), 3)]

s = "Find textbooks with titles containing 'NLP', " \
    "or 'natural' and 'language', or" \
    "'computational' and 'linguistics'."
len(set(s.split()))
np.arange(1, 12+1).prod()  # factorial (12) = arange(1, 13).prod()
#########################################################################################














