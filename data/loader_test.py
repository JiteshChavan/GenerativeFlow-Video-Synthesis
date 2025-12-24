from loader import make_wds
import json
labels = json.load(open("clips/labels.json"))
for key in sorted(labels, key=labels.get):
    print(labels[key],key)

b = next(iter(make_wds("shards_latent/000000.tar", batch_size=2, num_workers=0)))
print(b["z"].shape, b["z"].dtype, b["label_id"])
print(b['label_id'])
print ("done")