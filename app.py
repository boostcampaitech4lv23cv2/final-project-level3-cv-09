import streamlit as st
import yaml

import os
import cv2
import numpy as np
import re

st.set_page_config(layout="wide")

root = "../data/"
save_dir = os.path.join(root + "new")


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
        viz[idx] = tab_bbox.checkbox(cls + f"{idx}", value=True)
    else:
        viz[idx] = tab_bbox.checkbox(cls + f"{idx}", value=False)


def make_bbox(image, labels, viz, bar_columns):
    for idx, labels in enumerate(labels):

        labels = labels.split(" ")
        label, coordinate = labels[0], labels[1:]
        make_check_box(bar_columns, st.session_state.classes[int(label)], viz, idx)

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


def make_bbox_image(index, tab_bbox, bar_columns):
    image = cv2.imread(os.path.join(st.session_state.datas[index]), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    labels_txt = open(st.session_state.labels[index], "r")
    labels = labels_txt.readlines()
    viz = [True for _ in labels]
    labels_txt.close()

    image = make_bbox(image, labels, viz, bar_columns)
    image = cv2.resize(image, (960, 540))

    tab_bbox.image(image, caption="Selected Image")

    return labels, viz


def main():
    st.title("FindersAI")
    if "delete_list" not in st.session_state:
        (
            st.session_state.classes,
            st.session_state.datas,
            st.session_state.labels,
            st.session_state.filenames,
        ) = load_data()
        st.session_state.delete_list = ""

    idx = st.select_slider(
        "Select Image",
        range(len(st.session_state.filenames)),
        format_func=lambda x: st.session_state.filenames[x],
    )

    tab_bbox, tab_prediction = st.tabs(["Image", "Predict"])

    label_info, confirmed_label = make_bbox_image(idx, tab_bbox, st.sidebar)

    col1, col2 = tab_bbox.columns(2)

    delete_btn = col1.button(label="삭제 추가", key="delete_btn")
    cancel_delete_btn = col2.button(label="삭제 취소", key="delete_btn2")

    if delete_btn:
        st.session_state.delete_list += st.session_state.datas[idx] + "\n"

    if cancel_delete_btn:
        st.session_state.delete_list = re.sub(
            st.session_state.datas[idx] + "\n", "", st.session_state.delete_list
        )

    checkbox = st.sidebar.checkbox("재대로 됐는지 확인하세요")
    btn_clicked = st.sidebar.button(
        "BBox Save", key="confirm_btn", disabled=(checkbox is False)
    )

    st.sidebar.text_area(label="delete Image List", value=st.session_state.delete_list)
    btn_clicked2 = st.sidebar.button(
        "Make Delete Image txt file",
        key="confirm_btn2",
    )

    if btn_clicked:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_txt = [
            label for idx, label in enumerate(label_info) if confirmed_label[idx]
        ]
        f = open(
            os.path.join(save_dir, f"{st.session_state.labels[idx].split('/')[-1]}"),
            "w",
        )
        for l in save_txt:
            f.write(l)
        f.close()

    if btn_clicked2:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_txt2 = st.session_state.delete_list
        f = open(
            os.path.join(save_dir, "remove_list.txt"),
            "w",
        )
        f.write(save_txt2)
        f.close()


main()
