import json
import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import h5py
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


MODEL_PATH = "gemstone_model.h5"
CLASS_NAMES_PATH = "class_names.json"
IMAGE_PATH = "test1 .jpg"
IMG_SIZE = (224, 224)




def repair_legacy_lambda_model(model_path):
    with h5py.File(model_path, "r+") as f:
        model_config = f.attrs.get("model_config")
        if model_config is None:
            return

        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")

        config = json.loads(model_config)
        changed = False

        for layer in config["config"]["layers"]:
            if layer.get("class_name") != "Lambda":
                continue

            layer["class_name"] = "Rescaling"
            layer["config"] = {
                "name": layer.get("name", "mobilenetv2_preprocess"),
                "trainable": True,
                "dtype": layer["config"].get("dtype"),
                "scale": 1 / 127.5,
                "offset": -1,
            }

            for node in layer.get("inbound_nodes", []):
                if isinstance(node.get("kwargs"), dict):
                    node["kwargs"].pop("mask", None)

            changed = True

        if changed:
            f.attrs.modify("model_config", json.dumps(config).encode("utf-8"))


repair_legacy_lambda_model(MODEL_PATH)
model = load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

img_path = os.path.abspath(IMAGE_PATH)
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# The model already contains MobileNetV2 preprocessing, so pass raw 0-255 pixels.
prediction = model.predict(img_array, verbose=0)

if len(class_names) != prediction.shape[-1]:
    raise ValueError(
        f"Class name count ({len(class_names)}) does not match model output "
        f"count ({prediction.shape[-1]}). Re-train the model or update "
        f"{CLASS_NAMES_PATH}."
    )

predicted_index = int(np.argmax(prediction))
predicted_class = class_names[predicted_index]
confidence = float(np.max(prediction) * 100)

print(f"Gemstone   : {predicted_class}")
print(f"Confidence : {confidence:.2f}%")
print()
print("All class probabilities:")
for name, prob in zip(class_names, prediction[0]):
    bar = "#" * int(prob * 40)
    print(f"  {name:<18} {prob*100:6.2f}%  {bar}")
