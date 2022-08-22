# %%
import time
import torch
import torchvision

from src.datasets.coco.base import get_coco
import src.models.coco.utils as utils
from src.models.coco.engine import train_one_epoch
from src.models.coco.presets import DetectionPresetTrain

torch.cuda.empty_cache()
model_name = "fasterrcnn_resnet50_fpn"

NUM_CLASSES = 91
LR = 0.02
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LR_STEPS = [16, 22]
LR_GAMMA = 0.1
EPOCHS = 1
PRINT_FREQ = 20
NUM_WORKERS = 2

kwargs = {"trainable_backbone_layers": None}
model = torchvision.models.detection.__dict__[model_name](
    weights=None, weights_backbone=None, num_classes=NUM_CLASSES, **kwargs
)

parameters = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(
    parameters,
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    nesterov=False,  # TODO check
)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=LR_STEPS, gamma=LR_GAMMA
)

scaler = torch.cuda.amp.GradScaler()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# train_transforms = DetectionPresetTrain(data_augmentation="hflip")

ds_train = get_coco("train")
# ds_test = get_coco("val")

# %%
loader_train = torch.utils.data.DataLoader(
    ds_train, num_workers=NUM_WORKERS, collate_fn=utils.collate_fn, batch_size=4
)
# loader_test = torch.utils.data.DataLoader(
#         dataset, num_workers=NUM_WORKERS, collate_fn=utils.collate_fn
#     )

model.to(device)
start_time = time.time()
for epoch in range(EPOCHS):
    train_one_epoch(model, optimizer, loader_train, device, epoch, PRINT_FREQ, scaler)
    lr_scheduler.step()

    # evaluate after every epoch
    # evaluate(model, data_loader_test, device=device)

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print(f"Training time {total_time_str}")

# %%
