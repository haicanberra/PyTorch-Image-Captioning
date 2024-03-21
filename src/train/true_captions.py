import json, os

# Extract the 1st 1000 images with their captions for reference.

assert os.path.isfile('dataset\\nobug\\val\\captions_val2014.json')
counter = 1000
name = []
annotations = []
with open('dataset\\nobug\\val\\captions_val2014.json', 'r') as f:
    data = json.load(f)
    for a in data['images']:
        if (counter != 0):
            name.append(a['file_name'])
            for b in data['annotations']:
                num = b['image_id']
                string = "0"*(12-len(str(num))) + str(num)
                if string in a['file_name']:
                    annotations.append(b['caption'])
                    break
            counter = counter -1
        else:
            break

dictionary = {}
for i in range(len(name)):
    dictionary[name[i]] = [annotations[i]]
with open('output\\data.json', 'w') as f:
    json.dump(dictionary, f)
