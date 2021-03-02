import datetime
import os
import requests


def send_message(message):
    url = os.environ.get('MESSAGE_PUSH_URL')
    if url:
        url = f"{url}{message}"
        res = requests.get(url)
        if res.status_code != 200:
            print('Failed to send message.')


def get_datetime():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
