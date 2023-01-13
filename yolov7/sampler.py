import random

sample = 1700

f = open("/opt/ml/data/dataset/train_124.txt", "r")

session = {
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0,
    "6": 0,
    "7": 0,
    "8": 0,
    "9": 0,
    "10": 0,
    "11": 0,
    "12": 0,
    "3rd": 0,
}

folder = {
    "1": [],
    "2": [],
    "3": [],
    "4": [],
    "5": [],
    "6": [],
    "7": [],
    "8": [],
    "9": [],
    "10": [],
    "11": [],
    "12": [],
    "3rd": [],
}

cnt = 0
a = f.readlines()
# print(a)
# print(len(a))
for i in a:
    test = i[15:]
    cnt += 1
    if test[:3] == "3rd":
        session.update({"3rd": session["3rd"] + 1})
        index = "3rd"
        target = folder[index]
        target.append(i)
        folder.update({index: target})

    elif test[:2] == "10":
        session.update({"10": session["10"] + 1})
        index = "10"
        target = folder[index]
        target.append(i)
        folder.update({index: target})

    elif test[:2] == "11":
        session.update({"11": session["11"] + 1})
        index = "11"
        target = folder[index]
        target.append(i)
        folder.update({index: target})

    elif test[:2] == "12":
        session.update({"12": session["12"] + 1})
        index = "12"
        target = folder[index]
        target.append(i)
        folder.update({index: target})

    else:
        index = test[0]
        session.update({index: session[index] + 1})
        index = test[0]
        target = folder[index]
        target.append(i)
        folder.update({index: target})


f.close()


new_sample = []
for value in folder.values():
    new_sample.extend(random.sample(value, k=sample))

random.shuffle(new_sample)

print(new_sample)
print(len(new_sample))

with open("/opt/ml/train.txt", "w") as file:
    for s in new_sample:
        file.write(s)


print(folder)
print(session)
print(cnt)
# file.write("Hello~ \n")
# file.write("World!")
