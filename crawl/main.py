import tweepy
import json
import base64
import MySQLdb
import time
import sys
from envs import *


for i in range(5):
    try:
        conn = MySQLdb.connect(user=db_username, passwd=db_pass,
                               host=db_host, db="mysql", use_unicode=True, charset="utf8mb4")
        print("connect success.")
        break
    except:
        print("connect error to DB... Try again in 30 seconds.")
        time.sleep(30)
else:
    print("Error connecting DB. system stoped.")
    sys.exit(1)


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

query = "INSERT INTO d2.tweet VALUES (%(tweet_id)s, %(created_at)s, %(text)s, %(user_id)s, %(tweet_json)s)"
c = conn.cursor()

while True:
    try:
        for tweet in tweepy.Cursor(api.home_timeline).items():
            tweet_id = tweet.id
            created_at = tweet.created_at
            text = tweet.text
            user_id = tweet.user.id
            tweet_json = json.dumps(tweet._json, ensure_ascii=False)
            data = {"tweet_id": tweet_id, "created_at": str(
                created_at), "text": text, "user_id": user_id, "tweet_json": tweet_json}
            c.execute(query, data)
            conn.commit()
    except Exception as e:
        print(e)
        time.sleep(60*5)
