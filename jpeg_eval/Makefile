all: data

data: 
	sudo mount -o size=2G -t tmpfs none /mnt/tmpfs	
	cp -r /data/zhijing/flickrImageNetV2/matched_frequency_part/ /mnt/tmpfs/
	mkdir /mnt/tmpfs/sorted_cache/
clean:
	sudo umount /mnt/tmpfs	
