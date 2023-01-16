import random

sample = 10000

f = open("/opt/ml/data/dataset/train_124.txt", "r")

path = "/opt/ml/data/dataset/labels/"

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

# print(keys)
# print(type(keys))
# print(folder)
# print(len(folder))

# select session
standard = 4

for i in range(sample):
    keys = list(folder.keys())
    bg = random.sample(keys, k=1)[0]
    keys.remove(bg)

    bg_img = random.sample(folder[bg], k=1)[0]
    with open("/opt/ml/data/mixed/bg.txt", "a") as f:
        f.write(bg_img)
    f.close()
    mask = random.sample(keys, k=1)[0]
    mask_img = random.sample(folder[mask], k=1)[0]
    with open("/opt/ml/data/mixed/mask.txt", "a") as f:
        f.write(mask_img)
    f.close()
    bg_label = bg_img[15:-4] + "txt"
    mask_label = mask_img[15:-4] + "txt"

    print(bg, mask)
    with open(path + bg_label, "r") as f:
        bg_rl = f.readlines()
    f.close()

    with open(path + mask_label, "r") as f:
        mask_rl = f.readlines()
    f.close()
    # print(type(mask_rl))

    tot = bg_rl + mask_rl

    # print(bg_rl)
    # print(mask_rl)
    # print(tot)
    # 0000~9999 이런 식으로 나오기

    # i formating
    num = str(i)
    add_zero_num = standard - len(num)
    num = "0" * add_zero_num + num

    with open("/opt/ml/data/mixed/mixed_124.txt", "a") as f:
        f.write(f"mixed/images/mixed_{num}.jpg\n")
    f.close()

    with open(f"/opt/ml/data/mixed/labels/mixed_{num}.txt", "w") as f:
        for t in tot:
            f.write(t)
    f.close

    # print(bg_label)

    # print(bg, mask)
