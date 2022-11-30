import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

_auth = None


def auth(key):
    """
    authenticate to keg server

    @param key: keg api key
    """
    global _auth
    s = requests.Session()
    retries = Retry(total=5,
                    backoff_factor=0.1,
                    status_forcelist=[ 500, 502, 503, 504 ])
    s.mount('http://', HTTPAdapter(max_retries=retries))
    s.mount('https://', HTTPAdapter(max_retries=retries))
    try:
        response = s.get(f'https://pylab.co/keg/auth/?key={key}')
    except requests.exceptions.RequestException:
        raise RuntimeError(r'Can\'t not connect to keg server.')
    if response.status_code != 200:
        raise RuntimeError(r'Can\'t not connect to keg server.')
    _auth = response.json()
    return _auth


def log(message):
    """
    log to keg server

    @param message: log message
    """
    global _auth
    if not is_authenticated():
        raise RuntimeError(r'No authentication.')
    if 'active' not in _auth or not _auth['active']:
        raise RuntimeError(r'Invalid license.')
    requests.post('https://pylab.co/keg/log/', data={
        'key': _auth['id'],
        'text': message
    })


def deduct():
    """
    deduct api usage

    @rtype: bool
    @return: a flag indicating if api usage was successfully deducted
    """
    global _auth
    if not is_authenticated():
        raise RuntimeError(r'No authentication.')
    response = requests.post('https://pylab.co/keg/auth/', data={
        'key': _auth['id']
    })
    return response.status_code == 200


def is_authenticated():
    """
    the flag which indicates current session is authenticated

    @rtype: bool
    @return: a flag indicating if current session is authenticated
    """
    global _auth
    return _auth and 'id' in _auth
