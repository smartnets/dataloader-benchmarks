# %%
from src.utils.persist import get_s3_results, is_s3_up
from src.config import settings as st
from pathlib import Path


# %%
if is_s3_up():
    s3 = get_s3_results()
# %%
b = s3.Bucket(st.BUCKET_NAME)

# %%
for obj in b.objects.all():
    key = obj.key
    if key.endswith("parameters.json"):
        print(key)
        key = key.replace(st.BUCKET_NAME, st.DOWNLOADED_RESULTS_DIR)
        path = Path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        b.download_file(obj.key, str(path))

print("Finished downloading results from s3.")
# %%
