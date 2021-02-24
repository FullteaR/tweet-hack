FROM python:3.8.1

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y\
 sudo\
 wget\
 git\
 curl\
 graphviz\
 build-essential\
 aptitude\
 swig\
 cmake\
 graphviz\
 python3-dev\
 default-libmysqlclient-dev
 
 #install juman
 RUN git clone https://github.com/ku-nlp/jumanpp.git /jumanpp && mkdir /jumanpp/cmake-build-dir
 WORKDIR /jumanpp/cmake-build-dir
 RUN cmake ..
 RUN make
 
 RUN aptitude install mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file -y

RUN pip install setuptools\
 jupyter\
 pandas\
 gensim\
 seaborn\
 matplotlib\
 tqdm\
 sklearn\
 timeout_decorator\
 sqlalchemy\
 tweepy\
 mecab-python3==0.996.3\
 gensim\
 pyknp\
 mysqlclient\
 tensorflow==2.3.0\
 xgboost\
 bs4

WORKDIR /
RUN git clone https://github.com/sonoisa/sentence-transformers
WORKDIR sentence-transformers
RUN pip install --no-cache-dir -r requirements.txt
RUN python setup.py install


RUN mkdir /root/.jupyter && touch /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo c.NotebookApp.open_browser = False >> /root/.jupyter/jupyter_notebook_config.py
WORKDIR /
RUN wget -O sonobe-datasets-sentence-transformers-model.tar "https://www.floydhub.com/api/v1/resources/JLTtbaaK5dprnxoJtUbBbi?content=true&download=true&rename=sonobe-datasets-sentence-transformers-model-2"
RUN tar -xvf sonobe-datasets-sentence-transformers-model.tar

ADD credential.json /

WORKDIR /mnt
#CMD jupyter notebook --allow-root  --NotebookApp.token=''
CMD python notify.py
