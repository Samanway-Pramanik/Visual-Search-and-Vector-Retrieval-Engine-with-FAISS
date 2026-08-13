import tarfile 
import os 

# path to ABO dataset(.tar file)
tar_path = "../data/abo-images-small.tar" 

output_dir = "../data/images" 

os.makedirs(output_dir, exist_ok=True) 

# Exracting first 1k image from teh .tar file
with tarfile.open(tar_path, "r") as tar :
    count = 0
    
    for member in tar.getmembers() :
        if not member.isfile() :
            continue # non files skipped
        
        if count < 1000 :
            tar.extract(member, path=output_dir)  
            count += 1 
        
        else :
            break
        

print(f"Extracted first {count} images to {output_dir}") 
