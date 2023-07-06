from flask import Flask, request, jsonify, session

from utils.url_parse import url_to_id
from utils.youtube_transcript import get_captions
from utils.vdb import VDB
from utils.model import model
import uuid


app = Flask(__name__)

app.secret_key = 'BAD_SECRET_KEY'

vector_dbs = {}

#### For Development Only
@app.route('/storevideo', methods = ['POST', 'GET'])
def storevideo():
    args = request.get_json()
    url = args['url']

    #get video id from url
    video_id = url_to_id(url)

    #get captions
    captions = get_captions(video_id)
    captions_texts = []
    for caption in captions:
        captions_texts.append(caption['text'])

    #get embeddings
    embeddings = model.encode(captions_texts)

    #intialize VDB and store embeddings in it
    vector_db = VDB( len(captions), 50)
    vector_db.train(embeddings)
    vector_db.insert(embeddings, captions)

    #generate universally unique id 
    token  = str(uuid.uuid4())
    session["token"] = token
    vector_dbs[token] = vector_db
    print(vector_dbs)
    
    return jsonify(token)


@app.route('/search', methods = ['POST'])
def search():
    args = request.get_json()
    query = args['query']

    #get query embedding 
    embedding = model.encode(query)

    #get the VDB
    token = session.get("token", None)
    if token == None:
        return jsonify("no video sent to server")

    print(token)
    print(vector_dbs)
    vector_db = vector_dbs.get(token, None)
    if vector_db == None:
        return jsonify("vector database not found")
    
    #search in the VDB
    result = vector_db.search(embedding, 5)

    return jsonify(result)


@app.route('/removeindex', methods = ['DELETE'])
def removeindex():
    #get the VDB
    token = session.get("token", None)
    if token == None:
        return jsonify("no video sent to server")

    print(token)
    print(vector_dbs)
    del vector_dbs[token]
    del session["token"]
    print(vector_dbs)

    return jsonify("success")