from src.config import settings as st
from src.utils.general import config_to_bool

file_content = """[default]
access_key = {aws_key}
access_token = 
add_encoding_exts = 
add_headers = 
bucket_location = US
ca_certs_file = 
cache_file = 
check_ssl_certificate = True
check_ssl_hostname = True
cloudfront_host = cloudfront.amazonaws.com
connection_max_age = 5
connection_pooling = True
content_disposition = 
content_type = 
default_mime_type = binary/octet-stream
delay_updates = False
delete_after = False
delete_after_fetch = False
delete_removed = False
dry_run = False
enable_multipart = True
encoding = UTF-8
encrypt = False
expiry_date = 
expiry_days = 
expiry_prefix = 
follow_symlinks = False
force = False
get_continue = False
gpg_command = /usr/bin/gpg
gpg_decrypt = %(gpg_command)s -d --verbose --no-use-agent --batch --yes --passphrase-fd %(passphrase_fd)s -o %(output_file)s %(input_file)s
gpg_encrypt = %(gpg_command)s -c --verbose --no-use-agent --batch --yes --passphrase-fd %(passphrase_fd)s -o %(output_file)s %(input_file)s
gpg_passphrase = 
guess_mime_type = True
host_base = {my_s3_endpoint_without_https}
host_bucket = {my_bucket}.s3.amazonaws.com
human_readable_sizes = False
invalidate_default_index_on_cf = False
invalidate_default_index_root_on_cf = True
invalidate_on_cf = False
kms_key = 
limit = -1
limitrate = 0
list_md5 = False
log_target_prefix = 
long_listing = False
max_delete = -1
mime_type = 
multipart_chunk_size_mb = 15
multipart_copy_chunk_size_mb = 1024
multipart_max_chunks = 10000
preserve_attrs = True
progress_meter = True
proxy_host = 
proxy_port = 0
public_url_use_https = False
put_continue = False
recursive = False
recv_chunk = 65536
reduced_redundancy = False
requester_pays = False
restore_days = 1
restore_priority = Standard
secret_key = {aws_secret}
send_chunk = 65536
server_side_encryption = False
signature_v2 = False
signurl_use_https = False
simpledb_host = sdb.amazonaws.com
skip_existing = False
socket_timeout = 300
ssl_client_cert_file = 
ssl_client_key_file = 
stats = False
stop_on_error = False
storage_class = 
throttle_max = 100
upload_id = 
urlencoding_mode = normal
use_http_expect = False
use_https = {use_https}
use_mime_magic = True
verbosity = WARNING
website_endpoint = {my_s3_endpoint}
website_error = 
website_index = index.html
"""


def setup_s3cmd():

    is_aws = config_to_bool(st.IS_AWS)
    USE_HTTPS = is_aws
    S3_WITHOUT = st.s3_endpoint.split("//")[1] if not is_aws else "s3.amazonaws.com"
    ENDPOINT = (
        st.s3_endpoint
        if not is_aws
        else "http://%(bucket)s.s3-website-%(location)s.amazonaws.com/"
    )

    with open("/home/worker/.s3cfg", "w") as fh:
        fh.write(
            file_content.format(
                aws_key=st.aws_access_key_id,
                aws_secret=st.aws_secret_access_key,
                my_bucket=st.bucket_name,
                my_s3_endpoint=ENDPOINT,
                my_s3_endpoint_without_https=S3_WITHOUT,
                use_https=USE_HTTPS,
            )
        )
    return None


if __name__ == "__main__":

    setup_s3cmd()
