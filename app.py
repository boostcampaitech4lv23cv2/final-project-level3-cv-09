import streamlit as st
import yaml

import os
import cv2
import numpy as np
import copy

st.set_page_config(layout="wide")

root = "../data/"


@st.cache(allow_output_mutation=True)
def load_data():
    # text file 담당 부분 개별 생성 해야합니다.
    yaml_path = os.path.join(root, "test/4th_run_124.yaml")
    f = open(yaml_path, "r")
    dataset_info = yaml.load(f, yaml.FullLoader)
    txt_dir = dataset_info["test"]  # txt file_dir
    classes = {idx: name for idx, name in enumerate(dataset_info["names"])}
    f.close()

    data_dir_txt = open(txt_dir, "r")
    datas = [os.path.join(root, file[:-1]) for file in data_dir_txt.readlines()]
    labels = [file.replace("images", "labels").replace("jpg", "txt") for file in datas]
    file_names = [file_dir.split("/")[-1] for file_dir in datas]
    data_dir_txt.close()

    return classes, datas, labels, file_names


def make_check_box(tab_bbox, cls, viz, idx) -> None:
    if viz[idx]:
        viz[idx] = tab_bbox.checkbox(cls, value=True)
    else:
        viz[idx] = tab_bbox.checkbox(cls, value=False)


def make_bbox(image, labels, viz, tab_label):
    for idx, labels in enumerate(labels):

        labels = labels.split(" ")
        label, coordinate = labels[0], labels[1:]
        make_check_box(tab_label, st.session_state.classes[int(label)], viz, idx)

        if viz[idx]:
            centerX, centerY, width, height = map(float, coordinate)
            box = [0, 0, 0, 0]
            box[0] = (centerX - width / 2.0) * 1920
            box[1] = (centerY - height / 2.0) * 1080
            box[2] = (centerX + width / 2.0) * 1920
            box[3] = (centerY + height / 2.0) * 1080
            # box[:, 2] = box[:, 0] + box[:, 2]
            # box[:, 3] = box[:, 1] + box[:, 3]

            box = list(map(int, box))
            image = cv2.rectangle(
                image,
                (box[0], box[1]),
                (box[2], box[3]),
                (255, 0, 0),
                3,
            )

    return image


def make_bbox_image(index, tab_bbox):
    tab_bbox_col1, tab_bbox_col2 = tab_bbox.columns(2)

    image = cv2.imread(os.path.join(st.session_state.datas[index]), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    labels_txt = open(st.session_state.labels[index], "r")
    labels = labels_txt.readlines()
    viz = [True for _ in labels]
    labels_txt.close()

    image = make_bbox(image, labels, viz, tab_bbox_col2)
    # image = cv2.resize(image, (640, 640))

    tab_bbox_col1.image(image, caption="Selected Image")


def main():
    st.title("FindersAI")

    (
        st.session_state.classes,
        st.session_state.datas,
        st.session_state.labels,
        st.session_state.file_names,
    ) = load_data()

    idx = st.selectbox(
        "Select Image",
        range(len(st.session_state.datas)),
        format_func=lambda x: st.session_state.datas[x],
    )

    tab_bbox, tab_prediction = st.tabs(["Image", "Predict"])

    make_bbox_image(idx, tab_bbox)


main()
