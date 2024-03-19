import glob
import os
import cv2
from copy import deepcopy

if __name__ == "__main__":
    datapath = "WarpDoc"
    img_list = []
    _t = "val"
    ext = ["*.jpg", "*.png"]
    for e in ext:
        img_list.extend(glob.glob(f"{datapath}/{_t}/image/**/{e}", recursive=True))
    
    sep = os.sep

    for img in img_list:
        parts = img.split(sep)

        new_parts = deepcopy(parts)
        new_parts[-1] = new_parts[-1][:-4] + ".txt"
        new_parts[-3] = "label"
        new_parts[-4] = new_parts[-4] + "_label"

        label_path = sep.join(new_parts)
        with open(label_path, "r") as f:
            file = f.readlines()
            bbox = [list(map(int, line.rstrip('\n').split()[1:])) for line in file]

            if len(bbox) > 1:
                bbox = list(sorted(bbox, key=lambda x: (x[2]-x[0])*(x[3]-x[1])))
            
            if len(bbox) > 0:
                bbox = bbox[-1]
        
        save_parts = deepcopy(parts)
        save_parts[-3] = "image"
        save_parts[-4] = save_parts[-4] + "_cut"
        save_path = sep.join(save_parts)

        print(save_path)
    



         