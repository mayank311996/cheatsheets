from fastapi import FastAPI
from ml import nlp
from pydantic import BaseModel
from typing import List

app = FastAPI()


#######################################################################################
@app.get("/")
def read_main():
    return {"message": "Hello World"}


@app.get("/article/{article_id}")
def analyze_article(article_id: int, q: str = None):
    return {"article_id": article_id,
            "q": q}


# if you want to make "q: query parameter" as required then remove the default value (None in this case)


#######################################################################################
@app.get("/")
def read_main():
    return {"message": "Hello World"}


@app.get("/article/{article_id}")
def analyze_article(article_id: int, q: str = None):
    count = 0
    if q:
        doc = nlp(q)
        count = len(doc.ents)
    return {"article_id": article_id,
            "q": q,
            "count": count}


#########################################################################################
# now we don't want to pass data as query parameter, we want to pass more complex json stuff
# so, changing get method to post
@app.get("/")
def read_main():
    return {"message": "Hello World"}


@app.post("/article/")
def analyze_article(body: dict):
    return body


#########################################################################################
@app.get("/")
def read_main():
    return {"message": "Hello World"}


class Article(BaseModel):
    content: str


@app.post("/article/")
def analyze_article(article: Article):
    # return article
    return {"message": article.content}


#########################################################################################
@app.get("/")
def read_main():
    return {"message": "Hello World"}


class Article(BaseModel):
    content: str
    comments: list = []  # defining as empty list so it becomes non required parameter


@app.post("/article/")
def analyze_article(article: Article):
    # return article
    return {"message": article.content,
            "comments": article.comments}


#########################################################################################
@app.get("/")
def read_main():
    return {"message": "Hello World"}


class Article(BaseModel):
    content: str
    comments: List[str] = []  # defining as empty list so it becomes non required parameter


@app.post("/article/")
def analyze_article(article: Article):
    doc = nlp(article.content)
    ents = []
    for ent in doc.ents:
        ents.append({"text": ent.text,
                     "label": ent.label})
    return {"message": article.content,
            "comments": article.comments,
            "ents": ents}


#########################################################################################
# now say we want to process multiple articles at a time (as might be the case in real
# world scenario)
@app.get("/")
def read_main():
    return {"message": "Hello World"}


class Article(BaseModel):
    content: str
    comments: List[str] = []  # defining as empty list so it becomes non required parameter


@app.post("/article/")
def analyze_article(articles: List[Article]):
    """
    Analyze an article and extract entities with spaCy
    :param articles: JSON object
    :return: entities and comments
    """
    ents = []
    comments = []
    for article in articles:
        for comment in article.comments:
            comments.append(comment.upper())
        doc = nlp(article.content)
        for ent in doc.ents:
            ents.append({"text": ent.text,
                         "label": ent.label})
    return {"ents": ents,
            "comments": comments}



