
# tracking based image anotations

Use a video to automatically generate instance segmentations for a single object using labelme format

#### Installation

Follow the pysot install instructions [here](https://github.com/STVIR/pysot/blob/master/INSTALL.md)

#### Usage

```bash
./main.sh <label name> <output directory> <video name> <decimation factor>
```

Example: Generate annotations from a video named `video7.mp4`, label car, output directory ./dataset and a decimation factor of 10
applied to each generated mask
```bash
./main.sh car ./dataset video7.mp4 10
```

