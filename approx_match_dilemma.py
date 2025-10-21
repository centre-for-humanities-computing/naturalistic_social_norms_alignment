import torch, json, os
from sentence_transformers import SentenceTransformer

import re, argparse
from tqdm import tqdm


def chunkize(lst, n):
    """
    Splits a list into overlapping chunks of length n,
    with an overlap of n/2 between consecutive chunks.

    Example:
        chunkize([1, 2, 3, 4, 5, 6, 7], 4)
        -> [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7]]
    """
    step = n // 2  # overlap = n/2
    chunks = [lst[i:i + n] for i in range(0, len(lst), step) if i < len(lst)]
    return chunks


def transcript_sentence_split(lst, n):
    sentence_ends = "!?."

    sentences = []

    current_sentence = []
    for segment_dict in lst:

        current_sentence.append(segment_dict)

        if any([segment_dict["text"].endswith(end_s) for end_s in sentence_ends]):
            sentences.append(current_sentence)
            current_sentence = []

    if current_sentence and current_sentence not in sentences:
        sentences.append(current_sentence)

    sentences = chunkize(sentences, n)

    sentences = [
        sorted(sum(s, []), key=lambda x: x["start"]) for s in sentences]

    return sentences


class DilemmasEmbeddingMatcher:

    def __init__(self, embedding_model):
        self.embedder = SentenceTransformer(embedding_model)

        self.corpus = []
        self.corpus_meta = []

    def chunk_to_text(self, chunkized_segments):
        text = " ".join([x["text"] for x in chunkized_segments])
        text = re.sub(r'\s{2,}', ' ', text)

        return text

    def prepare_corpus(self, transcribed_data_segments, n_chunks=8):
        transcribed_data_segments = list(sorted(transcribed_data_segments, key=lambda x: x["start"]))

        chunkized_segments = transcript_sentence_split(transcribed_data_segments, n_chunks)

        # prepare text
        self.corpus = []
        self.corpus_meta = []

        for segments_chunk in chunkized_segments:
            self.corpus.append(self.chunk_to_text(segments_chunk))
            self.corpus_meta.append({"start": segments_chunk[0]["id"], "end": segments_chunk[-1]["id"]})

    def search_dilemmas(self, dilemmas, transcribed_data_segments, n_chunks=8, top_k=5):
        output = {}

        self.prepare_corpus(
            transcribed_data_segments=transcribed_data_segments,
            n_chunks=n_chunks
        )

        corpus_embeddings = self.embedder.encode_document(self.corpus, convert_to_tensor=True)

        for query in dilemmas:
            query_embedding = self.embedder.encode_query(query, convert_to_tensor=True)
            similarity_scores = self.embedder.similarity(query_embedding, corpus_embeddings)[0]
            scores, indices = torch.topk(similarity_scores, k=top_k)

            output[query] = {f"top_{top_k}_similarities": [], "selected": None}
            for i in indices:
                data_dict = self.corpus_meta[i]
                data_dict["similarity"] = similarity_scores[i].cpu().item()
                output[query][f"top_{top_k}_similarities"].append(data_dict)
            output[query]["selected"] = min(output[query][f"top_{top_k}_similarities"], key=lambda x: x["start"])

        return output


def run(args):
    matcher = DilemmasEmbeddingMatcher(args.embedding_model)

    perfile_matched_dilemmas = {}

    selected_filenames = json.load(open(args.filtered_filenames_json, "r"))

    parsed_dilemmas = json.load(open(args.parsed_descriptions_json, "r"))

    for filename in tqdm(selected_filenames):
        matched_dilemmas = matcher.search_dilemmas(
            dilemmas=parsed_dilemmas[filename]["dilemmas"],
            transcribed_data_segments=json.load(
                open(
                    os.path.join(args.transcript_loaddir, filename), "r"
                )
            )["segments"],
            n_chunks=args.n_chunks,
            top_k=args.top_k,
        )
        perfile_matched_dilemmas[filename] = matched_dilemmas

    with open(args.savepath, "w") as f:
        json.dump(perfile_matched_dilemmas, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search for the approximate first occurrence of the dilemma based on the embedding similarity"
    )

    parser.add_argument(
        "--transcript_loaddir",
        type=str,
        default="SpeakerDiarizedTranscripts-SaraOgMonopolet",
        help="path to dir, which contains transcript json files (with ot without speakers)",
    )

    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model to use.",
    )

    parser.add_argument(
        "--n_chunks",
        type=int,
        default=3,
        help="Number of sentences to use for the emdedding similarity search. n_chunks sentences will be compared to the dilemma text to see if this dilemma appeared in this chunk.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Out of top_k most similar chunks, select the earliest one to find an approximate position of the dilemma",
    )

    parser.add_argument(
        "--filtered_filenames_json",
        type=str,
        default="filtered_meta/filtered_filenames.json",
        help="Path to JSON file containing a list of selected JSON filenames to process.",
    )

    parser.add_argument(
        "--parsed_descriptions_json",
        type=str,
        default="filtered_meta/reviewed_parsed_descriptions.json",
        help="Path to JSON file mapping filenames to parsed descriptions.",
    )


    parser.add_argument(
        "--savepath",
        type=str,
        default="ApproxDilemmasSegmentLocation.json",
        help="save path, to where save the output json.",
    )



    return parser.parse_args()

if __name__ == '__main__':
    run(parse_args())


