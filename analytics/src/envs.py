import json

infos = json.load(open("../credential.json", "r"))

consumer_key = infos["consumer_key"]
consumer_secret = infos["consumer_secret"]
access_token = infos["access_token"]
access_token_secret = infos["access_token_secret"]

db_username = "root"
db_pass = "root"
db_host = "d2_db"
