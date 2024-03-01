# hankook_tire

# Setting
- install requirements
```
pip install -r requirements.txt
```
- Download data(zip, csv) to data file
- data/ in Footprint_DB_Refine_v2.csv
- data/ in Footshape_Gray_Image_All.zip

- unzip settings
```
apt-get update
apt install unzip
apt install zip
```

- unzip to data/Footprint/ in Footprint image
```
unzip data/Gray_Image_for_AI.zip -d data/Footprint/
unzip data/Footprint_latent.zip -d data/
```
<br />

# Multi-GPU command
```
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 main.py
```

# Setting Git
```
git config --global user.email "dybroh@kaist.ac.kr"
git config --global user.name "Noverse0"
```
<br />