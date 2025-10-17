import torch
import numpy as np
import librosa
from pyannote.audio import Inference
from sklearn.cluster import AgglomerativeClustering
import os
import json
from tqdm import tqdm
from pyannote.audio import Model

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torchaudio
import multiprocessing as mp
import argparse


class SpeakerAssignment:

    @staticmethod
    def get_intersection(s_0, e_0, s_1, e_1):
        if  e_0 < s_1 or e_1 < s_0:
            return 0

        return min(e_0, e_1) - max(s_0, s_1)



    @staticmethod
    def get_speaker(predicted_speaker_segments, transcribed_segment):

        biggest_intersection, speakername = 0, None

        for speaker_segment in predicted_speaker_segments:
            if speaker_segment["start"] > transcribed_segment["end"]:
                break
            else:

                intersection = SpeakerAssignment.get_intersection(
                        s_0=speaker_segment["start"],
                        e_0=speaker_segment["end"],
                        s_1=transcribed_segment["start"],
                        e_1=transcribed_segment["end"]
                )
                if intersection > biggest_intersection:
                    biggest_intersection = intersection
                    speakername = speaker_segment["speaker"]

        return speakername

    @staticmethod
    def assign_speaker_to_texts(predicted_speaker_segments, transcribed_segments):

        for i in range(len(transcribed_segments)):
            transcribed_segments[i]["speaker"] = SpeakerAssignment.get_speaker(
                predicted_speaker_segments=predicted_speaker_segments, transcribed_segment=transcribed_segments[i]
            )

        return transcribed_segments


def diarize_file(wav_filepath, json_filepath, description_dict, savedir, device, use_auth_token):
    print(f">>>> Device {device}")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=use_auth_token
    ).to(torch.device(device))
    print(f"> PPL loaded")

    try:
        n_participants = len(description_dict["parsed_monopolet"])
    except:
        n_participants = 25

    if not n_participants:
        max_speakers = 25
    else:
        max_speakers = n_participants + 5


    with ProgressHook() as hook:
        print("> Loading file")
        waveform, sample_rate = torchaudio.load(wav_filepath)
        print("> Loaded file")
        print("Starting speaker diarization...")
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, max_speakers=max_speakers, hook=hook)

    print("---> DONE speaker diarization...")

    predicted_speaker_segments = []
    for segment_tuple in list(diarization.exclusive_speaker_diarization):
        predicted_speaker_segments.append({
            "speaker": segment_tuple[1],
            "start": segment_tuple[0].start,
            "end": segment_tuple[0].end,
            "duration": segment_tuple[0].duration
        })
    predicted_speaker_segments = list(sorted(predicted_speaker_segments, key=lambda x: x["start"]))

    print("+++ DONE diarization parsing...")

    transcribed_data = json.load(open(json_filepath, "r"))
    transcribed_segments = transcribed_data["segments"]
    transcribed_segments = list(sorted(transcribed_segments, key=lambda x: x["start"]))

    transcribed_speaker_segments = SpeakerAssignment.assign_speaker_to_texts(
        predicted_speaker_segments=predicted_speaker_segments, transcribed_segments=transcribed_segments
    )

    print("+++ DONE speaker assignment; Saving ...")

    transcribed_data["segments"] = transcribed_speaker_segments
    with open(os.path.join(
            savedir, os.path.basename(wav_filepath).replace(".wav", ".json")
    ), "w") as f:
        json.dump(transcribed_data, f)


def get_wav_fiepath(wav_dir, json_filepath):
    return os.path.join(wav_dir, json_filepath.replace(".json", ".wav"))

def run(args):
    if args.num_gpus == -1:
        num_gpus = torch.cuda.device_count()

    elif args.num_gpus == 0:
        num_gpus = 0

    else:
        num_gpus = args.num_gpus


    if num_gpus:
        devices = [f"cuda:{i}" for i in range(num_gpus)]
    else:
        devices = ["cpu"]


    os.makedirs(args.savedir, exist_ok=True)

    selected_files = json.load(open(args.filtered_filenames_json, "r"))
    descriptions = json.load(open(args.parsed_descriptions_json, "r"))

    os.makedirs(args.savedir, exist_ok=True)

    print("------- TOTAL FILES: ", len(selected_files))
    tasks = []
    i = 0
    for json_filepath in selected_files:
        wav_filepath = get_wav_fiepath(args.wav_dir, json_filepath)
        if os.path.basename(wav_filepath).replace(".wav", ".json") not in os.listdir(args.savedir):
            if json_filepath in descriptions:
                tasks.append((
                    wav_filepath,
                    os.path.join(args.transcripts_path, json_filepath),
                    descriptions[json_filepath],
                    args.savedir,
                    devices[i % len(devices)]
                ))
                i += 1

    print("------- LEFT TO PROCESS FILES: ", len(tasks))

    n_processes = min(args.n_per_device * num_gpus, mp.cpu_count() - 2)

    print(f"> Starting {n_processes} processes")

    results = []
    mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=n_processes) as pool:
        for args in tqdm(tasks, desc="Submitting tasks"):
            results.append(pool.apply_async(diarize_file, args))
        results = [result.get() for result in results]

        pool.close()
        pool.join()

    print("===== All done! =====")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel diarization pipeline using multiple GPUs or CPUs."
    )

    parser.add_argument(
        "--wav_dir",
        type=str,
        required=True,
        help="Directory containing WAV files corresponding to JSON metadata files.",
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
        "--transcripts_path",
        type=str,
        default="Transcript -- hviske-v3-conversation -- sara-og-monopolet",
        help="Directory containing transcript files referenced by JSON metadata.",
    )

    parser.add_argument(
        "--savedir",
        type=str,
        default="SpeakerDiarizedTranscripts-SaraOgMonopolet",
        help="Directory where diarization output files will be saved.",
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help=(
            "Number of GPUs to use. "
            "-1 = use all available GPUs, "
            "0 = use CPU only, "
            "N = use N GPUs."
        ),
    )

    parser.add_argument(
        "--n_per_device",
        type=int,
        default=2,
        help="Number of parallel processes to launch per GPU (or per CPU if num_gpus=0).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)


