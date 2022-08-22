# %%
import pandas as pd
from torch_tb_profiler.plugin import TorchProfilerPlugin

# %%
class Context:
    def __init__(self, l):
        self.logdir = l


# %%
# path = "./results/cifar10/webdataset"
path = "./results/coco/hub"
c = Context(path)


# %%
tp = TorchProfilerPlugin(c)

# %%
view = "Overview"
run_name = "traces"
run = tp._get_run(run_name)
workers = run.get_workers(view)
# %%
for worker in workers:
    spans = run.get_spans(worker)
    for span in spans:

        profile = tp._get_profile(run_name, worker, span)
        overview = profile.overview
# %%

df = pd.DataFrame(overview["performance"][0]["children"])
df.index = df.name
# %%
df.plot.pie(y="value")
# %%
