from binance.client import Client
import configparser
from binance.websockets import BinanceSocketManager
from twisted.internet import reactor
from time import sleep

# Loading keys from config file
config = configparser.ConfigParser()
config.read_file(open('/home/venom/GitHub/cheatsheets/medium/algo_trading/binance_API/secret.cfg'))
api_key = config.get('BINANCE', 'API_KEY')
secret_key = config.get('BINANCE', 'SECRET_KEY')


client = Client(api_key, secret_key)


client.API_URL = 'https://testnet.binance.vision/api'


btc_price = {'error':False}

def btc_trade_history(msg):
    """ define how to process incoming WebSocket messages """
    print("Inside function")
    if msg['e'] != 'error':
        print(msg['c'])
        btc_price['last'] = msg['c']
        btc_price['bid'] = msg['b']
        btc_price['ask'] = msg['a']
    else:
        btc_price['error'] = True



# init and start the WebSocket
bsm = BinanceSocketManager(client)
conn_key = bsm.start_symbol_ticker_socket('BTCUSDT', btc_trade_history)
bsm.start()

sleep(60)


# stop websocket
bsm.stop_socket(conn_key)

# properly terminate WebSocket
reactor.stop()