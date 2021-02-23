FROM mysql:5.7

ARG MYSQL_ROOT_PASSWORD=root

ENV MYSQL_ROOT_PASSWORD ${MYSQL_ROOT_PASSWORD}
ENV TZ Asia/Tokyo

ADD my.cnf /etc/mysql/conf.d

CMD ["mysqld"]
