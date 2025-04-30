import json
import random

import numpy as np

from itertools import groupby
from toolz import partition_all, first, second

import streamlit as st


def read_file(path, parse_line=lambda line: line.strip()):
    with open(path, "r") as f:
        return list(map(parse_line, f.readlines()))


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


class Dataset:
    def __init__(self):
        concepts, norms, norm_to_concepts, norm_to_id = self.load_norms()

        self.concepts = concepts
        self.norms = norms
        self.norm_to_concepts = norm_to_concepts
        self.norm_to_id = norm_to_id

        self.concept_to_image_name = dict(
            read_file(
                "data/things-image-names.txt",
                lambda line: line.split(),
            )
        )
        self.context_sentences = self.load_context_sentences()

    def load_norms(self, num_min_concepts=10):
        concepts_and_norms = read_json("data/mcrae-x-things.json")
        concepts_and_norms = sorted(concepts_and_norms, key=second)
        concepts = sorted(set(c for c, _ in concepts_and_norms))

        norm_groups = groupby(concepts_and_norms, key=second)
        norm_to_concepts = {
            norm: list(map(first, group)) for norm, group in norm_groups
        }

        norms = sorted(norm_to_concepts.keys())
        norm_to_id = {norm: i for i, norm in enumerate(norms)}

        norms = [
            norm
            for norm, concepts in norm_to_concepts.items()
            if len(concepts) >= num_min_concepts
        ]
        return concepts, norms, norm_to_concepts, norm_to_id

    def get_image_path(self, concept):
        IMG_URL = "https://things-initiative.org/uploads/THINGS/images_resized/{}/{}"
        return IMG_URL.format(concept, self.concept_to_image_name[concept])

    def load_context_sentences(self):
        def remove_starting_number(sentence):
            sentence = sentence.strip()
            num, *words = sentence.split()
            num = num.replace(".", "")
            assert num.isdigit()
            return " ".join(words)

        def parse_line(line):
            data = json.loads(line)
            id_ = data["id"]
            sentences = data["response"]
            sentences = sentences.split("\n")
            sentences = [s for s in sentences if s]
            sentences = [remove_starting_number(s) for s in sentences]
            return id_, sentences

        path = f"data/gpt4o_concept_context_sentences.jsonl"
        return dict(read_file(path, parse_line))

    def get_context_sentences(self, concept):
        return self.context_sentences[concept]


MODELS = [
    # Vision-only
    "random-siglip",
    "vit-mae-large",
    "max-vit-large",
    "max-vit-large-in21k",
    "swin-v2-ssl",
    "dino-v2",
    # Vision and language
    "siglip-224",
    "pali-gemma-224",
    "clip",
    # Language-only
    "glove-840b-300d-word",
    "numberbatch-word",
    "clip-word",
    "gemma-2b-contextual-layers-9-to-18-seq-last-word",
]


def load_predictions(model_name):
    data = np.load("data/predictions-mcrae-x-things.npz")
    preds = data["results"]
    models = data["models"]
    models = models.tolist()
    i = models.index(model_name)
    return preds[:, i]


st.set_page_config(layout="wide", page_title="THINGS Dataset")


def main():
    dataset = Dataset()

    ORDER_FUNCS = {
        "random": lambda xs, n: random.sample(range(len(xs)), n),
        "score": lambda xs, n: np.argsort(xs)[-n:][::-1],
    }

    with st.sidebar:
        norm = st.selectbox("Norm", dataset.norms)
        num_to_show = st.number_input("Num. to show", value=12, step=4)
        model_name = st.selectbox("Model", MODELS, index=MODELS.index("clip"))
        order = st.selectbox("Order", ORDER_FUNCS.keys(), index=1)

    num_cols = 4
    predictions = load_predictions(model_name)

    num_norms, num_concepts = predictions.shape
    assert num_norms == len(dataset.norms)
    assert num_concepts == len(dataset.concepts)

    predictions = predictions[dataset.norms.index(norm)]
    indices = ORDER_FUNCS[order](predictions, num_to_show)

    for group in partition_all(num_cols, indices):
        cols = st.columns(num_cols)
        for col, i in zip(cols, group):
            concept = dataset.concepts[i]
            pred = predictions[i]
            context_texts = dataset.get_context_sentences(concept)
            has_norm = concept in dataset.norm_to_concepts[norm]
            has_norm_str = "✓" if has_norm else "✗"
            lines = [
                "concept: {}".format(concept),
                "has norm: {}".format(has_norm_str),
                "score: {:.3f}".format(pred),
            ]
            col.markdown("```\n{}\n```".format("\n".join(lines)))
            col.image(dataset.get_image_path(concept), caption=context_texts[0])


if __name__ == "__main__":
    main()
