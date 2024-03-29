#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import MySQLdb
import pandas as pd
import MeCab
from sklearn.preprocessing import LabelEncoder
import subprocess
import numpy as np
from tqdm.auto import tqdm
import re
import tweepy
import json
import time
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import pickle


# In[ ]:


infos = json.load(open("/credential.json", "r"))

consumer_key = infos["consumer_key"]
consumer_secret = infos["consumer_secret"]
access_token = infos["access_token"]
access_token_secret = infos["access_token_secret"]


# In[ ]:


con = MySQLdb.connect(user="root", passwd="root",
                               host="d2_db", db="mysql", use_unicode=True, charset="utf8mb4")

#cmd = 'echo `mecab-config --dicdir`"/mecab-ipadic-neologd"'
#path = (subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                           shell=True).communicate()[0]).decode('utf-8')
m = MeCab.Tagger("-Owakati")#.format(path))
model_path = "/training_bert_japanese"
model = SentenceTransformer(model_path, show_progress_bar=False)
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
tqdm.pandas()


# In[ ]:


df = pd.read_sql("SELECT * FROM d2.tweet", con=con)


# In[ ]:


iblank = re.compile("^[\s\t]*\n$")
reply_to = re.compile("@[0-9a-zA-Z_]+ ")
url = re.compile("https?://[0-9a-zA-Z-._~?&=/]+")

# https://crimnut.hateblo.jp/entry/2018/08/25/202253からhan2zen,zen2hanをコピーさせていただきました
ASCII_ZENKAKU_CHARS = (
    u'ａ', u'ｂ', u'ｃ', u'ｄ', u'ｅ', u'ｆ', u'ｇ', u'ｈ', u'ｉ', u'ｊ', u'ｋ',
    u'ｌ', u'ｍ', u'ｎ', u'ｏ', u'ｐ', u'ｑ', u'ｒ', u'ｓ', u'ｔ', u'ｕ', u'ｖ',
    u'ｗ', u'ｘ', u'ｙ', u'ｚ',
    u'Ａ', u'Ｂ', u'Ｃ', u'Ｄ', u'Ｅ', u'Ｆ', u'Ｇ', u'Ｈ', u'Ｉ', u'Ｊ', u'Ｋ',
    u'Ｌ', u'Ｍ', u'Ｎ', u'Ｏ', u'Ｐ', u'Ｑ', u'Ｒ', u'Ｓ', u'Ｔ', u'Ｕ', u'Ｖ',
    u'Ｗ', u'Ｘ', u'Ｙ', u'Ｚ',
    u'！', u'”', u'＃', u'＄', u'％', u'＆', u'’', u'（', u'）', u'＊', u'＋',
    u'，', u'－', u'．', u'／', u'：', u'；', u'＜', u'＝', u'＞', u'？', u'＠',
    u'［', u'￥', u'］', u'＾', u'＿', u'‘', u'｛', u'｜', u'｝', u'～', u'　'
)

ASCII_HANKAKU_CHARS = (
    u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k',
    u'l', u'm', u'n', u'o', u'p', u'q', u'r', u's', u't', u'u', u'v',
    u'w', u'x', u'y', u'z',
    u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H', u'I', u'J', u'K',
    u'L', u'M', u'N', u'O', u'P', u'Q', u'R', u'S', u'T', u'U', u'V',
    u'W', u'X', u'Y', u'Z',
    u'!', u'"', u'#', u'$', u'%', u'&', u'\'', u'(', u')', u'*', u'+',
    u',', u'-', u'.', u'/', u':', u';', u'<', u'=', u'>', u'?', u'@',
    u'[', u"¥", u']', u'^', u'_', u'`', u'{', u'|', u'}', u'~', u' '
)

KANA_ZENKAKU_CHARS = (
    u'ア', u'イ', u'ウ', u'エ', u'オ', u'カ', u'キ', u'ク', u'ケ', u'コ',
    u'サ', u'シ', u'ス', u'セ', u'ソ', u'タ', u'チ', u'ツ', u'テ', u'ト',
    u'ナ', u'ニ', u'ヌ', u'ネ', u'ノ', u'ハ', u'ヒ', u'フ', u'ヘ', u'ホ',
    u'マ', u'ミ', u'ム', u'メ', u'モ', u'ヤ', u'ユ', u'ヨ',
    u'ラ', u'リ', u'ル', u'レ', u'ロ', u'ワ', u'ヲ', u'ン',
    u'ァ', u'ィ', u'ゥ', u'ェ', u'ォ', u'ッ', u'ャ', u'ュ', u'ョ',
    u'。', u'、', u'・', u'゛', u'゜', u'「', u'」', u'ー'
)

KANA_HANKAKU_CHARS = (
    u'ｱ', u'ｲ', u'ｳ', u'ｴ', u'ｵ', u'ｶ', u'ｷ', u'ｸ', u'ｹ', u'ｺ',
    u'ｻ', u'ｼ', u'ｽ', u'ｾ', u'ｿ', u'ﾀ', u'ﾁ', u'ﾂ', u'ﾃ', u'ﾄ',
    u'ﾅ', u'ﾆ', u'ﾇ', u'ﾈ', u'ﾉ', u'ﾊ', u'ﾋ', u'ﾌ', u'ﾍ', u'ﾎ',
    u'ﾏ', u'ﾐ', u'ﾑ', u'ﾒ', u'ﾓ', u'ﾔ', u'ﾕ', u'ﾖ',
    u'ﾗ', u'ﾘ', u'ﾙ', u'ﾚ', u'ﾛ', u'ﾜ', u'ｦ', u'ﾝ',
    u'ｧ', u'ｨ', u'ｩ', u'ｪ', u'ｫ', u'ｯ', u'ｬ', u'ｭ', u'ｮ',
    u'｡', u'､', u'･', u'ﾞ', u'ﾟ', u'｢', u'｣', u'ｰ'
)

DIGIT_ZENKAKU_CHARS = (
    u'０', u'１', u'２', u'３', u'４', u'５', u'６', u'７', u'８', u'９'
)

DIGIT_HANKAKU_CHARS = (
    u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9'
)

KANA_TEN_MAP = (
    (u'ガ', u'ｶ'), (u'ギ', u'ｷ'), (u'グ', u'ｸ'), (u'ゲ', u'ｹ'), (u'ゴ', u'ｺ'),
    (u'ザ', u'ｻ'), (u'ジ', u'ｼ'), (u'ズ', u'ｽ'), (u'ゼ', u'ｾ'), (u'ゾ', u'ｿ'),
    (u'ダ', u'ﾀ'), (u'ヂ', u'ﾁ'), (u'ヅ', u'ﾂ'), (u'デ', u'ﾃ'), (u'ド', u'ﾄ'),
    (u'バ', u'ﾊ'), (u'ビ', u'ﾋ'), (u'ブ', u'ﾌ'), (u'ベ', u'ﾍ'), (u'ボ', u'ﾎ'),
    (u'ヴ', u'ｳ')
)

KANA_MARU_MAP = (
    (u'パ', u'ﾊ'), (u'ピ', u'ﾋ'), (u'プ', u'ﾌ'), (u'ペ', u'ﾍ'), (u'ポ', u'ﾎ')
)


ascii_zh_table = {}
ascii_hz_table = {}
kana_zh_table = {}
kana_hz_table = {}
digit_zh_table = {}
digit_hz_table = {}

for (az, ah) in zip(ASCII_ZENKAKU_CHARS, ASCII_HANKAKU_CHARS):
    ascii_zh_table[az] = ah
    ascii_hz_table[ah] = az

for (kz, kh) in zip(KANA_ZENKAKU_CHARS, KANA_HANKAKU_CHARS):
    kana_zh_table[kz] = kh
    kana_hz_table[kh] = kz

for (dz, dh) in zip(DIGIT_ZENKAKU_CHARS, DIGIT_HANKAKU_CHARS):
    digit_zh_table[dz] = dh
    digit_hz_table[dh] = dz

kana_ten_zh_table = {}
kana_ten_hz_table = {}
kana_maru_zh_table = {}
kana_maru_hz_table = {}
for (ktz, kth) in KANA_TEN_MAP:
    kana_ten_zh_table[ktz] = kth
    kana_ten_hz_table[kth] = ktz

for (kmz, kmh) in KANA_MARU_MAP:
    kana_maru_zh_table[kmz] = kmh
    kana_maru_hz_table[kmh] = kmz

del ASCII_ZENKAKU_CHARS, ASCII_HANKAKU_CHARS,     KANA_ZENKAKU_CHARS, KANA_HANKAKU_CHARS,    DIGIT_ZENKAKU_CHARS, DIGIT_HANKAKU_CHARS,    KANA_TEN_MAP, KANA_MARU_MAP

kakko_zh_table = {
    u'｟': u'⦅', u'｠': u'⦆',
    u'『': u'｢', u'』': u'｣',
    u'〚': u'⟦', u'〛': u'⟧',
    u'〔': u'❲', u'〕': u'❳',
    u'〘': u'⟬', u'〙': u'⟭',
    u'《': u'⟪', u'》': u'⟫',
    u'【': u'(', u'】': u')',
    u'〖': u'(', u'〗': u')'
}
kakko_hz_table = {}
for k, v in kakko_zh_table.items():
    kakko_hz_table[v] = k


def zen2han(text="", ascii_=True, digit=True, kana=True, kakko=True, ignore=()):
    result = []
    for c in text:
        if c in ignore:
            result.append(c)
        elif ascii_ and (c in ascii_zh_table):
            result.append(ascii_zh_table[c])
        elif digit and (c in digit_zh_table):
            result.append(digit_zh_table[c])
        elif kana and (c in kana_zh_table):
            result.append(kana_zh_table[c])
        elif kana and (c in kana_ten_zh_table):
            result.append(kana_ten_zh_table[c] + u'ﾞ')
        elif kana and (c in kana_maru_zh_table):
            result.append(kana_maru_zh_table[c] + u'ﾟ')
        elif kakko and (c in kakko_zh_table):
            result.append(kakko_zh_table[c])
        else:
            result.append(c)
    return "".join(result)


def han2zen(text, ascii_=True, digit=True, kana=True, kakko=True, ignore=()):
    result = []
    for i, c in enumerate(text):
        if c == u'ﾞ' or c == u'ﾟ':
            continue
        elif c in ignore:
            result.append(c)
        elif ascii_ and (c in ascii_hz_table):
            result.append(ascii_hz_table[c])
        elif digit and (c in digit_hz_table):
            result.append(digit_hz_table[c])
        elif kana and (c in kana_ten_hz_table) and (text[i + 1] == u'ﾞ'):
            result.append(kana_ten_hz_table[c])
        elif kana and (c in kana_maru_hz_table) and (text[i + 1] == u'ﾟ'):
            result.append(kana_maru_hz_table[c])
        elif kana and (c in kana_hz_table):
            result.append(kana_hz_table[c])
        elif kakko and (c in kakko_hz_table):
            result.append(kakko_hz_table[c])
        else:
            result.append(c)
    return "".join(result)


# In[ ]:


def process(tweet):
    if tweet[:2]=="RT":
        return ""
    try:
        tweet = reply_to.sub("@", tweet)
    except:
        pass
    try:
        tweet = url.sub("", tweet)
    except:
        pass
    try:
        tweet = re.sub("[！!]+", "!", tweet)
    except:
        pass
    try:
        tweet = re.sub("[？?]+", "?", tweet)
    except:
        pass
    try:
        tweet = han2zen(tweet, ascii_=False, digit=False,
                        kakko=False, kana=True)
    except:
        pass
    try:
        tweet = zen2han(tweet, kana=False)
    except:
        pass
    try:
        tweet = re.sub("\d+", "0", tweet)
    except:
        pass
    try:
        tweet = re.sub("\n+","\n",tweet)
    except:
        pass
    try:
        tweet = re.sub("\s+"," ",tweet)
    except:
        pass
    
    try:
        tweet = re.sub("[^ぁ-んァ-ヶー一-龠\x00-\x7F]+","",tweet)
    except:
        pass

    return tweet


# In[ ]:


df["text"] = df["text"].progress_apply(process)


# In[ ]:


df_favs = pd.DataFrame(columns=["tweet_id","created_at","text","user_id","tweet_json"])


# In[ ]:


while True:
    try:
        for tweet in tweepy.Cursor(api.favorites).items():
            tweet_id = tweet.id
            created_at = tweet.created_at
            print(created_at)
            text = tweet.text
            user_id = tweet.user.id
            tweet_json = json.dumps(tweet._json, ensure_ascii=False)
            s = pd.Series([tweet_id, created_at, text, user_id, tweet_json], index=df_favs.columns)
            df_favs = df_favs.append(s, ignore_index=True)
    except Exception as e:
        print(e)
        time.sleep(60*5)


# In[ ]:


df["favs"] = df["tweet_id"].progress_apply(lambda w: w in df_favs.tweet_id)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'vectors = np.asarray(model.encode(df.text))')


# In[ ]:


bst = xgb.XGBRegressor(n_estimators=100, n_jobs = -1)


# In[ ]:


bst.fit(vectors, df["favs"])


# In[ ]:


pickle.dump(bst, open("/model/bst_fav.pickle","wb"))

