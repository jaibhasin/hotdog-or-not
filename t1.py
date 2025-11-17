import os

# folder_path = "/Users/jaibhasin/Desktop/hot dog dataset/test/hot_dog" # 250
# folder_path = "/Users/jaibhasin/Desktop/hot dog dataset/test/not_hot_dog" # 250

# folder_path = "/Users/jaibhasin/Desktop/hot dog dataset/train/hot_dog˝" # 249

# folder_path = "/Users/jaibhasin/Desktop/hot dog dataset/train/not_hot_dog" # 249

# folder_path = "/Users/jaibhasin/Desktop/hot dog dataset/hotdog-nothotdog/hotdog-nothotdog/hotdog-nothotdog/test/hotdog" #200
# folder_path = "/Users/jaibhasin/Desktop/hot dog dataset/hotdog-nothotdog/hotdog-nothotdog/hotdog-nothotdog/test/nothotdog" #200

# folder_path = "/Users/jaibhasin/Desktop/hot dog dataset/hotdog-nothotdog/hotdog-nothotdog/hotdog-nothotdog/train/hotdog" # 2121
# folder_path = "/Users/jaibhasin/Desktop/hot dog dataset/hotdog-nothotdog/hotdog-nothotdog/hotdog-nothotdog/train/nothotdog" #2121

# folder_path = "/Users/jaibhasin/Desktop/hot dog dataset/test/hotdog" #2121

folder_path = "/Users/jaibhasin/Desktop/hot dog dataset/train/nothotdog"
count = len([f for f in os.listdir(folder_path) 
             if os.path.isfile(os.path.join(folder_path, f))])
print(count)



