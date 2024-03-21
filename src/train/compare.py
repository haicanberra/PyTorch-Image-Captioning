import json, os
from nltk import word_tokenize, meteor
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
import numpy as np

# Generate side by side comparison for true captions and predicted.
def compare():
    if not os.path.isfile('output\\compare.json'):
        jsonfile = json.load(open('output\\data.json','r'))
        with open('output\\output.txt', 'r') as file:
            # Iterate through each line in the file
            for line in file:
                # Split the line into two strings based on the comma
                parts = line.strip().split(',')
                jsonfile[parts[0]].append(parts[1])
            with open('output\\compare.json', "w") as outfile:
                json.dump(jsonfile, outfile)
    else:
        # NOTE: hypothesis = predicted using model
        #       reference = human written
        with open('output\\compare.json', 'r') as file:
            data = json.load(file)
            real_captions = []
            predicted_captions = []
            for key in data:
                real_captions.append(data[key][0])
                predicted_captions.append(data[key][1])
            real_captions = [word_tokenize(c.lower()) for c in real_captions]
            predicted_captions = [word_tokenize(c.lower()) for c in predicted_captions]

            meteor_score = []
            for i in range(len(real_captions)):
                meteor_score.append(round(meteor([real_captions[i]], predicted_captions[i]),4))
            
            bleu_score = []
            for i in range(len(real_captions)):
                bleu_score.append(sentence_bleu([real_captions[i]], predicted_captions[i]))
            
            plt.figure(figsize=(12, 5))
            bins = np.arange(0, 1, 0.1)
            # Plot the first subplot (score)
            plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
            plt.hist(meteor_score, bins=bins, edgecolor='k', alpha=0.7)
            plt.xlabel('METEOR Score Range')
            plt.ylabel('Number of Instances')
            plt.title("Average METEOR Score: "+str(round(sum(meteor_score)/len(meteor_score),3)))
            plt.xticks(np.arange(0, 1, 0.1))
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Plot the second subplot (score2)
            plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
            plt.hist(bleu_score, bins=bins, edgecolor='k', alpha=0.7)
            plt.xlabel('BLEU Score Range')
            plt.ylabel('Number of Instances')
            plt.title("Average BLEU Score: "+str(round(sum(bleu_score)/len(bleu_score),3)))
            plt.xticks(np.arange(0, 1, 0.1))
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Adjust layout spacing
            plt.tight_layout()
            plt.savefig("output\\score_plot.png")
            # Show the plot
            plt.show()

if __name__ == "__main__":
    # compare()
    with open('output\\compare.json', 'r') as file:
        data = json.load(file)
        real_captions = []
        predicted_captions = []
        filenames = ["COCO_val2014_000000483108.jpg", "COCO_val2014_000000337264.jpg", "COCO_val2014_000000542145.jpg", "COCO_val2014_000000235006.jpg", "COCO_val2014_000000026942.jpg"]
        for name in filenames:
            real_captions.append(data[name][0])
            predicted_captions.append(data[name][1])
        real_captions = [word_tokenize(c.lower()) for c in real_captions]
        predicted_captions = [word_tokenize(c.lower()) for c in predicted_captions]
        meteor_score = []
        for i in range(len(real_captions)):
            meteor_score.append(round(meteor([real_captions[i]], predicted_captions[i]),4))
        bleu_score = []
        for i in range(len(real_captions)):
            bleu_score.append(sentence_bleu([real_captions[i]], predicted_captions[i]))
    for i in range(len(real_captions)):
        print("METEOR: " + str(round(meteor_score[i],5)) + " / " + "BLEU: " + str(bleu_score[i]))

        
    



