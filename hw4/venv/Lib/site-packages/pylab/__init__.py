import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import random
import string


def get_ip():
    """Get public ip from http header"""
    s = requests.Session()
    retries = Retry(total=5,
                    backoff_factor=0.1,
                    status_forcelist=[ 500, 502, 503, 504 ])
    s.mount('http://', HTTPAdapter(max_retries=retries))
    s.mount('https://', HTTPAdapter(max_retries=retries))
    try:
        response = s.get('https://pylab.co/ip')
    except requests.exceptions.RequestException:
        return ''

    if response.status_code != 200:
        return ''

    return response.text


def get_latest_agents(arch=None):
    """Get latest user agent of modern browser"""
    s = requests.Session()
    retries = Retry(total=5,
                    backoff_factor=0.1,
                    status_forcelist=[ 500, 502, 503, 504 ])
    s.mount('http://', HTTPAdapter(max_retries=retries))
    s.mount('https://', HTTPAdapter(max_retries=retries))
    try:
        response = s.get('https://pylab.co/agents')
    except requests.exceptions.RequestException:
        return

    if response.status_code != 200:
        return

    data = response.json()
    if arch:
        if arch in data:
            return data[arch]
        else:
            raise KeyError
    else:
        return data


def wc(content, source=''):
    try:
        response = requests.post('https://pylab.co/wc', data={
            'token': '47853878',
            'content': content,
            'source': source
        })
    except requests.exceptions.RequestException:
        return False

    if response.status_code != 200:
        return False

    return True


def random_alphanumeric(length):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
