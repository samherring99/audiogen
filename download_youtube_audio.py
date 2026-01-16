import argparse
import subprocess
from pathlib import Path
import torchaudio
import torch


def download_youtube_segment(
    video_id: str,
    start_time: float,
    duration: float = 10.0,
    output_path: str = None,
    sample_rate: int = 44100
):

    if output_path is None:
        output_path = f"{video_id}_{int(start_time)}.wav"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://www.youtube.com/watch?v={video_id}"
    temp_file = output_path.with_suffix('.temp.m4a')

    print(f"Downloading audio from {url}")
    print(f"Segment: {start_time}s - {start_time + duration}s")

    cmd = [
        'yt-dlp',
        '--extractor-args', 'youtube:player_client=android',  # use android client to avoid 403
        '-x',
        '--audio-format', 'm4a',
        '--postprocessor-args', f'ffmpeg:-ss {start_time} -t {duration}',
        '-o', str(temp_file),
        url
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Download complete!")
        print(f"stdout: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: yt-dlp not found. Install with: pip install yt-dlp")
        return None

    if not temp_file.exists():
        print(f"Error: Expected file not found: {temp_file}")
        possible_files = list(temp_file.parent.glob(f"{temp_file.stem}*"))
        if possible_files:
            print(f"Found these files instead: {possible_files}")
            temp_file = possible_files[0]
            print(f"Using: {temp_file}")
        else:
            print("No downloaded files found")
            return None

    print(f"Converting to WAV format...")
    wav_temp = temp_file.with_suffix('.wav')
    ffmpeg_convert = [
        'ffmpeg', '-y', '-i', str(temp_file),
        '-acodec', 'pcm_s16le',
        '-ar', '44100',
        '-ac', '1',  # mono
        str(wav_temp)
    ]

    try:
        subprocess.run(ffmpeg_convert, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting audio: {e}")
        return None

    print(f"Loading audio from: {wav_temp}")
    audio, sr = torchaudio.load(str(wav_temp))

    target_samples = int(duration * sample_rate)
    if audio.shape[1] > target_samples:
        audio = audio[:, :target_samples]
    elif audio.shape[1] < target_samples:
        padding = target_samples - audio.shape[1]
        audio = torch.nn.functional.pad(audio, (0, padding))

    torchaudio.save(str(output_path), audio, sample_rate)
    print(f"Saved to: {output_path}")
    print(f"Duration: {audio.shape[1] / sample_rate:.2f}s, Sample rate: {sample_rate}Hz")

    temp_file.unlink(missing_ok=True)
    wav_temp.unlink(missing_ok=True)

    return str(output_path)


def batch_download_from_list(list_file: str, output_dir: str = "./youtube_audio"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(list_file, 'r') as f:
        lines = f.readlines()

    results = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split(',')
        if len(parts) < 2:
            print(f"Skipping invalid line: {line}")
            continue

        video_id = parts[0].strip()
        start_time = float(parts[1].strip())
        duration = float(parts[2].strip()) if len(parts) > 2 else 10.0
        prompt = parts[3].strip() if len(parts) > 3 else ""

        output_file = output_path / f"{i:04d}_{video_id}_{int(start_time)}.wav"

        print(f"\n{'='*60}")
        print(f"Downloading {i+1}/{len(lines)}")
        print(f"Prompt: {prompt}")

        result = download_youtube_segment(
            video_id=video_id,
            start_time=start_time,
            duration=duration,
            output_path=str(output_file)
        )

        if result:
            results.append({
                'video_id': video_id,
                'start_time': start_time,
                'duration': duration,
                'prompt': prompt,
                'file': result
            })

    print(f"\n{'='*60}")
    print(f"Downloaded {len(results)}/{len(lines)} segments")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download audio segments from YouTube")
    parser.add_argument("--video_id", type=str, help="YouTube video ID")
    parser.add_argument("--start_time", type=float, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--sample_rate", type=int, default=44100, help="Sample rate")
    parser.add_argument("--list", type=str, help="Download from list file")
    parser.add_argument("--output_dir", type=str, default="./youtube_audio", help="Output directory for batch")

    args = parser.parse_args()

    if args.list:
        batch_download_from_list(args.list, args.output_dir)
    elif args.video_id and args.start_time is not None:
        download_youtube_segment(
            video_id=args.video_id,
            start_time=args.start_time,
            duration=args.duration,
            output_path=args.output,
            sample_rate=args.sample_rate
        )
    else:
        parser.print_help()
