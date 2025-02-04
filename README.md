# Kaist Dataset Relabeling Tools

This repo contains a set of tools to update labels from Kaist dataset along with the updated labels.

For more information about the dataset go check its webpage: [KAIST Multispectral Pedestrian Detection Benchmark [CVPR '15]](https://soonminhwang.github.io/rgbt-ped-detection/).

You can use the following command to download the dataset. The path should match this repository root so that the `annotations-xml-new` of the downloaded data is substituted by the one contained in this repo leaving the images in its corresponding place.

```sh
echo "Downloading dataset to ${GIT_REPO_ROOT}"
filename="kaist-cvpr15.tar.gz"
url="https://onedrive.live.com/download?cid=1570430EADF56512&resid=1570430EADF56512%21109419&authkey=AJcMP-7Yp86PWoE"
wget --no-check-certificate ${url} -O ${GIT_REPO_ROOT}/${filename}

echo "Extract dataset (takes > 10 mins)"
tar zxvf ${GIT_REPO_ROOT}/${filename} -C ${GIT_REPO_ROOT}
```

## Scripts included in the respsitory

- [ ] `src/01_parseYoloLabels.py`: Parses and store in a dict-format all YOLO output labels from the configured folder.
- [ ] `src/02_processFalsePositives.py`: From a given cache file of YOLO labels it compares against KAIST labels to search for False Positives.
- [ ] `src/03_checkFalsePositives.py`: It checks each false positive projected over the image for the user to accept the incoming label or not.