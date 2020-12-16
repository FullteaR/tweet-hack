FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN apt update && apt upgrade -y && apt install -y\
 wget\
 git\
 curl\
 build-essential\
 cmake\
 graphviz
 
 #install juman
 RUN git clone https://github.com/ku-nlp/jumanpp.git /jumanpp && mkdir /jumanpp/cmake-build-dir
 WORKDIR /jumanpp/cmake-build-dir
 RUN cmake ..
 RUN make

RUN pip install --upgrade setuptools pip
RUN pip install\
 opencv-python\
 seaborn\
 matplotlib\
 tqdm\
 sklearn\
 timeout_decorator\
 jupyter\
 gensim\
 pyknp\
 tensorflow==2.3.0

RUN git clone https://github.com/sonoisa/sentence-transformers
WORKDIR sentence-transformers
RUN pip install -r requirements.txt
RUN python setup.py install


RUN mkdir /root/.jupyter && touch /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo c.NotebookApp.open_browser = False >> /root/.jupyter/jupyter_notebook_config.py
RUN wget -O sonobe-datasets-sentence-transformers-model.tar "https://www.floydhub.com/api/v1/resources/JLTtbaaK5dprnxoJtUbBbi?content=true&download=true&rename=sonobe-datasets-sentence-transformers-model-2"
RUN tar -xvf sonobe-datasets-sentence-transformers-model.tar


WORKDIR /mnt
CMD jupyter notebook --allow-root  --NotebookApp.token=''
