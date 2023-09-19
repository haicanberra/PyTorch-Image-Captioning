import json

# temp = [57870, 384029, 222016, 520950]
temp = [184613, 318219, 391895, 522418]
with open("dataset\\nobug\\val\\annotations\\captions_val2014.json", "r") as file:
    data = json.load(file)
    for key in data["annotations"]:
        if key["image_id"] in temp:
            print(key)
