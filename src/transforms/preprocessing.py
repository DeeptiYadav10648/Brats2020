from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandSpatialCropd,
    EnsureTyped,
    MapTransform
)

class RemapLabels(MapTransform):

    def __call__(self, data):

        d = dict(data)

        label = d["label"]

        label[label == 4] = 3

        d["label"] = label

        return d


train_transform = Compose([

    LoadImaged(
        keys=["image", "label"]
    ),

    EnsureChannelFirstd(
        keys=["image", "label"]
    ),

    RemapLabels(
        keys=["label"]
    ),

    NormalizeIntensityd(
        keys="image",
        nonzero=True,
        channel_wise=True
    ),

    RandSpatialCropd(
        keys=["image", "label"],
        roi_size=(128,128,128),
        random_size=False
    ),

    EnsureTyped(
        keys=["image", "label"]
    )
])


val_transform = Compose([

    LoadImaged(
        keys=["image", "label"]
    ),

    EnsureChannelFirstd(
        keys=["image", "label"]
    ),

    RemapLabels(
        keys=["label"]
    ),

    NormalizeIntensityd(
        keys="image",
        nonzero=True,
        channel_wise=True
    ),

    EnsureTyped(
        keys=["image", "label"]
    )
])


test_transform = Compose([
    LoadImaged(
        keys=["image", "label"]
    ),

    EnsureChannelFirstd(
        keys=["image", "label"]
    ),

    RemapLabels(
        keys=["label"]
    ),

    NormalizeIntensityd(
        keys="image",
        nonzero=True,
        channel_wise=True
    ),

    EnsureTyped(
        keys=["image", "label"]
    )
])