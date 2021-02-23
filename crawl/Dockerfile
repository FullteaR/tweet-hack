FROM python:3.8.1

RUN pip install mysqlclient tweepy

ADD crawl/*.py /
ADD credential.json /

CMD python main.py