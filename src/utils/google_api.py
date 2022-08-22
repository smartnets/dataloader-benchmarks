import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from src.config import settings as st


def get_client():
    scopes = ["https://spreadsheets.google.com/feeds"]
    creds_dict = json.loads(st.google_json[1:-1])
    creds_dict["private_key"] = creds_dict["private_key"].replace("\\\\n", "\n")
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scopes)

    client = gspread.authorize(creds)
    return client


def upload_results(data):

    client = get_client()
    spreadsheet = client.open_by_url(st.sheet_url)
    sheet = spreadsheet.worksheet("raw")

    ids = sheet.col_values(1)  # Get Users Id
    if len(ids) == 0:
        sheet.append_row(list(data.keys()))

    values = []
    for v in data.values():
        if isinstance(v, list):
            v_ = ",".join(map(str, v))
        else:
            v_ = v
        values.append(v_)
    sheet.append_row(values)
