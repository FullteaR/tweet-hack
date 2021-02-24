# sns-hack

## REQUIREMENTS
- docker-compose
- Docker

## INSTALL
### Twitter apiの入手
https://developer.twitter.com/en/apps/ の右上にあるCreate an appを押して、アプリケーションを作成します。

API key, API secret, Access Token, Access Token Secretを控えておきます。
### LINE apiの入手
https://developers.line.biz/consoleにログインして、新しいchannelを作成します。

Channnel access tokenとYour user IDを控えておきます。

### 起動
控えたIDを、credential.jsonに書き込みます。

```docker-compose up -d```を入力します。
