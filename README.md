
# tracking based image anotations

Use a video to automatically generate instance segmentations for a single object using labelme format

#### Installation

```bash
git clone --recursive https://github.com/travis575757/tracking_based_image_annotations
cd tracking_based_image_annotations
cd pysot
bash install.sh <conda install directory> pysot
conda activate pysot
pip install gdown
cd ..
./main.sh <label name> <output directory> <video name> <decimation factor>
```
#### Usage

```bash
./main.sh <label name> <output directory> <video name> <decimation factor>
```

Example: Generate annotations from a video named `video7.mp4`, label car, output directory ./dataset and a decimation factor of 10
applied to each generated mask.  All maskes are generated based on a polygon drawn by the user on the first video frame.
```bash
./main.sh car ./dataset video7.mp4 10
```

