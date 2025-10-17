import os
import multiprocessing as mp
from tqdm import tqdm
import torch
import subprocess
import argparse


def transcribe_worker(savedir, modeldir, filename, device):
    print(">> DEVICE ", device)

    if device != "cpu":

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]  # e.g. "0", "1", etc.

        cmd = [
            "whisper",
            filename,
            "--model_dir", modeldir,
            "--device", "cuda",
            "--output_dir", savedir,
            "--language", "Danish"
        ]
    else:
        cmd = [
            "whisper",
            filename,
            "--model_dir", modeldir,
            "--output_dir", savedir,
            "--language", "Danish"
        ]
    # Run the command and wait for it to finish
    subprocess.run(cmd, check=True, env=env)

    return filename


def run(args):

    if args.num_gpus == -1:
        num_gpus = torch.cuda.device_count()

    elif args.num_gpus == 0:
        num_gpus = 0

    else:
        num_gpus = args.num_gpus

    os.makedirs(args.savedir, exist_ok=True)

    SUPPORTED_EXTS = ["mp3", "wav", "flac", "m4a", "ogg"]
    audio_files_names = [filename for filename in os.listdir(args.audio_dir) if filename.split(".")[-1] in SUPPORTED_EXTS]
    print("Number of audio files:", len(audio_files_names))

    # check if already processed
    audio_files = [
        os.path.join(args.audio_dir, x) for x in audio_files_names if
        x.replace(".mp3", ".json") not in os.listdir(args.savedir)
    ]

    print("Number of left audio files:", len(audio_files))

    if num_gpus:
        devices = [f"cuda:{i}" for i in range(num_gpus)]
    else:
        devices = ["cpu"]

    n_processes = args.n_per_device * num_gpus


    tasks = [(
        args.savedir,
        args.modeldir,
        filename,
        devices[i % len(devices)]
    ) for i, filename in enumerate(audio_files)]

    results = []
    with mp.Pool(processes=n_processes) as pool:
        for args in tqdm(tasks, desc="Submitting tasks"):
            results.append(pool.apply_async(transcribe_worker, args))
        pool.close()
        pool.join()


    print("All transcriptions completed.")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch transcribe audio files using multiple GPUs or CPUs."
    )

    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing audio files to process (supported: mp3, wav, flac, m4a, ogg).",
    )

    parser.add_argument(
        "--savedir",
        type=str,
        required=True,
        help="Directory where transcription results will be saved.",
    )

    parser.add_argument(
        "--modeldir",
        type=str,
        required=True,
        help="Directory containing the whisper model or model checkpoint files; Or Whisper model e.g. \"medium\".",
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
        help="Number of parallel processes per GPU (or per CPU if num_gpus=0).",
    )

    args = parser.parse_args()
    run(args)




