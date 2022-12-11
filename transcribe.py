import argparse
import torch
from transformers import pipeline


def transcribe(filename, model_id, language):
    device = 0 if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        chunk_length_s=30,
        device=device,
    )

    pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")
    text = pipe(filename)["text"]

    model_name = f"{MODEL_NAME}".replace("/", "_")
    text_file = file.replace(".mp3", "") + f"{model_name}.txt"
    with open(text_file, 'w') as f:
        f.write(text)
    print(f"Wrote file '{text_file}'")


def read_parameters():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "filename",
        help="MP3 file to transcribe",
        type=argparse.FileType('r'))

    parser.add_argument(
        "--model_id",
        type=str,
        default="softcatala/whisper-small-ca",
        help="Model identifier",
    )

    parser.add_argument(
        "--language",
        type=str,
        default="ca",
        help="Two letter language code for the transcription language",
    )

    args = parser.parse_args()
    return args.filename, args.model_id, args.language

if __name__ == "__main__":
    filename, model_id, language = read_parameters()
    

