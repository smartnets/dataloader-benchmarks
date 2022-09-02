import datetime
from pathlib import Path
import boto3
import botocore
from botocore.client import Config
from botocore.client import ClientError
from src.config import settings as st


def folder_exists_and_not_empty(bucket: str, path: str) -> bool:
    """
    Folder should exists.
    Folder should not be empty.
    """
    creds = get_s3_creds()
    s3 = boto3.client(
        "s3",
        **creds,
        config=Config(signature_version="s3v4"),
        region_name=st.s3_region,
    )
    if not path.endswith("/"):
        path = path + "/"

    resp = s3.list_objects_v2(Bucket=bucket, Prefix=path, Delimiter="/", MaxKeys=1)
    return "CommonPrefixes" in resp


def is_s3_up():

    s3 = get_s3_results()
    try:
        s3.meta.client.head_bucket(Bucket=st.bucket_name)
        return True
    except ClientError:
        print("Bucket unavailable")
        return False
    except botocore.exceptions.EndpointConnectionError:
        print("Bucket unavailable")
        return False
        # The bucket does not exist or you have no access.


def get_s3_creds():
    return {
        "aws_access_key_id": st.AWS_ACCESS_KEY_ID,
        "aws_secret_access_key": st.AWS_SECRET_ACCESS_KEY,
        "endpoint_url": st.S3_ENDPOINT if st.S3_ENDPOINT != "None" else None,
    }


def get_s3_results():

    creds = get_s3_creds()
    s3 = boto3.resource(
        "s3",
        **creds,
        config=Config(signature_version="s3v4"),
        region_name=st.s3_region,
    )
    return s3


def save_results_s3(local_file, remote_file):

    s3 = get_s3_results()
    s3.Bucket(st.bucket_name).upload_file(local_file, remote_file)


def persist_results(local_path):

    exp_path = str(local_path).replace(st.local_results_dir, "")[1:]
    remote_dir = Path(st.bucket_name) / exp_path

    results_dir = Path(local_path)
    for local_res in results_dir.rglob("*.*"):
        s3_path = remote_dir / local_res.name
        save_results_s3(str(local_res), str(s3_path))
