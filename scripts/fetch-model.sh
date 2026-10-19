#!/usr/bin/env sh
# Download the released model into model-service/model/ so the service image can be built.
set -eu

VERSION="${MODEL_VERSION:-model-v1}"
BASE="https://github.com/mgorgiev/sms-phishing-detector/releases/download/${VERSION}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/model-service/model"

mkdir -p "$DEST"
for f in model.onnx tokenizer.json labels.json; do
  curl -fsSL --retry 3 -o "$DEST/$f" "$BASE/$f"
done
echo "fetched ${VERSION} into ${DEST}"
