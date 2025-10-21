import pandas as pd
import itertools
import plotly.express as px
import streamlit as st
import datetime as dt
from datetime import datetime, timedelta
from entsoe.files import EntsoeFileClient
from copy import deepcopy
import json


client = EntsoeFileClient(username=st.secrets["TransparencyPlatform"]["tp_user"], pwd=st.secrets["TransparencyPlatform"]["tp_passwd"])

file_list = client.list_folder('AcceptedAggregatedOffers_17.1.D')
df = client.download_single_file(folder='AcceptedAggregatedOffers_17.1.D', filename=list(file_list.keys())[0])

df = client.download_single_file(folder='TP_export', filename="Export_log_r3,csv")



def list_folder(p_client, folder: str) -> dict:
    """
    returns a dictionary of filename: unique file id
    """
    if not folder.endswith('/'):
        folder += '/'
    r = p_client.session.post(p_client.BASEURL + "listFolder",
                            data=json.dumps({
                                "path": "/TP_export/" + folder,
                                "sorterList": [
                                    {
                                        "key": "lastUpdatedTimestamp",
                                        "ascending": True
                                    }
                                ],
                                "pageInfo": {
                                    "pageIndex": 0,
                                    "pageSize": 5000  # this should be enough for basically anything right now
                                }
                            }),
                            headers={
                                'Authorization': f'Bearer {p_client.access_token}',
                                'Content-Type': 'application/json'
                            },
                            proxies=p_client.proxies, timeout=p_client.timeout)
    #r.raise_for_status()
    data = r.json()
    filtered = {
            x['name']: x['fileId']
            for x in data['contentItemList']
            if datetime.fromisoformat(x['lastUpdatedTimestamp'].replace('Z', '+00:00')) > p_last_stored
        }
    return filtered

x = list_folder(client, 'AcceptedAggregatedOffers_17.1.D' )