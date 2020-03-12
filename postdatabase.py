import requests
import json

API_ENDPOINT = "http://192.168.4.58:8888/api/v1/record"

def postdata(data, api = API_ENDPOINT):
    '''
    data struct
    data = {    "storeId": string (5e6222b469a7d4123bd7e062, 5e622306c8a6d0123b3919e6),
                "time": timestamp,
                "count": number
            }
    '''
    headers = {"Content-Type": "application/json"}
    r = requests.post(url=API_ENDPOINT, data=data, headers=headers)
    pastebin = r.text
    print("[INFO] the pastebin URL is {}".format(pastebin))