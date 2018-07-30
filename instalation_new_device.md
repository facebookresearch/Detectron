Copy files


0.Create dataset folder in mnt:
mkdir /mnt/2tb/detectron
mkdir /mnt/2tb/detectron/datasets 


1. Coco files:

rsync -av --progress -e "ssh -p 12022 -T -o Compression=no -x" elnazavr@office.rossum.ai:/mnt/3tb/detectron/datasets/coco 

2. VOC files

rsync -av --progress -e "ssh -p 12022 -T -o Compression=no -x" elnazavr@office.rossum.ai:/mnt/3tb/detectron/datasets/pascal_voc

4. COCOVOC files

rsync -av --progress -e "ssh -p 12022 -T -o Compression=no -x" elnazavr@office.rossum.ai:/mnt/3tb/detectron/datasets/cocovoc/*.json

5. Create symlinks from coco and voc

cd /mnt/3tb/detectron/datasets/cocovoc/
mkdir images 
cd images
for file in /mnt/2tb/detectron/datasets/coco/train2017/*; do  ln -s "$file" .; done
ln -sfn /mnt/2tb/detectron/datasets/cpascal_voc/VOCdevkit/VOC2012/JPEGImages/* .

6. Make symlinks to Detectron/lib/datasets/data

ln -s /mnt/2tb/detectron/datasets/coco/ .
ln -s /mnt/2tb/detectron/datasets/pascal_voc/ .
ln -s /mnt/2tb/detectron/datasets/cocovoc/ .

Hope that everything works...
