import json
import random
from typing import List


MUSIC_GENRES = [
    "electronic", "jazz", "classical", "rock", "hip hop", "ambient", "techno",
    "house", "trance", "dubstep", "lo-fi", "folk", "country", "blues", "reggae",
    "metal", "punk", "indie", "pop", "r&b", "soul", "funk", "disco"
]

MUSIC_MOODS = [
    "upbeat", "melancholic", "energetic", "calm", "dark", "bright", "dreamy",
    "aggressive", "peaceful", "nostalgic", "mysterious", "joyful", "sad", "intense"
]

MUSIC_INSTRUMENTS = [
    "piano", "guitar", "drums", "bass", "violin", "saxophone", "synthesizer",
    "flute", "trumpet", "cello", "harmonica", "accordion", "harp", "organ"
]

MUSIC_SETTINGS = [
    "in a concert hall", "in a small cafe", "in a studio", "at a festival",
    "in a church", "in a jazz club", "in a bedroom", "on a street corner"
]

ANIMALS = [
    "dog barking", "cat meowing", "bird chirping", "owl hooting", "wolf howling",
    "horse neighing", "cow mooing", "rooster crowing", "cricket chirping",
    "frog croaking", "lion roaring", "elephant trumpeting", "dolphin clicking"
]

NATURE_SOUNDS = [
    "rain falling", "thunder rumbling", "wind blowing", "waves crashing",
    "waterfall flowing", "stream babbling", "leaves rustling", "fire crackling",
    "snow crunching", "ice cracking"
]

URBAN_SOUNDS = [
    "car engine", "traffic noise", "train passing", "airplane flying",
    "construction work", "crowd chatter", "footsteps", "door closing",
    "phone ringing", "keyboard typing", "glass breaking", "siren wailing"
]

HOUSEHOLD_SOUNDS = [
    "washing machine running", "vacuum cleaner", "microwave beeping",
    "clock ticking", "water dripping", "doorbell ringing", "chair scraping",
    "dishes clattering", "keys jingling"
]

# Ambient environments
ENVIRONMENTS = [
    "forest", "beach", "city street", "mountain", "desert", "jungle",
    "park", "library", "restaurant", "gym", "office", "subway station",
    "airport", "hospital", "school", "market"
]

WEATHER = [
    "rainy day", "sunny afternoon", "stormy night", "windy morning",
    "foggy evening", "snowy day", "thunderstorm"
]

TIMES = [
    "morning", "afternoon", "evening", "night", "dawn", "dusk", "midnight"
]


def generate_music_prompt() -> str:
    templates = [
        f"{random.choice(MUSIC_MOODS)} {random.choice(MUSIC_GENRES)} music",
        f"{random.choice(MUSIC_INSTRUMENTS)} playing {random.choice(MUSIC_GENRES)} music",
        f"{random.choice(MUSIC_GENRES)} {random.choice(MUSIC_INSTRUMENTS)} {random.choice(MUSIC_SETTINGS)}",
        f"{random.choice(MUSIC_MOODS)} {random.choice(MUSIC_INSTRUMENTS)} melody",
        f"{random.choice(MUSIC_GENRES)} track with {random.choice(MUSIC_INSTRUMENTS)}",
    ]
    return random.choice(templates)


def generate_sound_effect_prompt() -> str:
    sound_categories = [ANIMALS, NATURE_SOUNDS, URBAN_SOUNDS, HOUSEHOLD_SOUNDS]
    sound = random.choice(random.choice(sound_categories))

    templates = [
        f"{sound}",
        f"{sound} loudly",
        f"{sound} softly",
        f"the sound of {sound}",
        f"{sound} repeatedly",
    ]
    return random.choice(templates)


def generate_ambient_prompt() -> str:
    templates = [
        f"ambient sounds of a {random.choice(ENVIRONMENTS)}",
        f"{random.choice(ENVIRONMENTS)} on a {random.choice(WEATHER)}",
        f"{random.choice(ENVIRONMENTS)} at {random.choice(TIMES)}",
        f"{random.choice(WEATHER)} in the {random.choice(ENVIRONMENTS)}",
        f"peaceful {random.choice(TIMES)} in a {random.choice(ENVIRONMENTS)}",
    ]
    return random.choice(templates)


def generate_complex_prompt() -> str:
    templates = [
        f"{random.choice(ANIMALS)} with {random.choice(NATURE_SOUNDS)} in the background",
        f"{random.choice(MUSIC_INSTRUMENTS)} playing while {random.choice(URBAN_SOUNDS)}",
        f"{random.choice(NATURE_SOUNDS)} and {random.choice(ANIMALS)} in a {random.choice(ENVIRONMENTS)}",
    ]
    return random.choice(templates)


def generate_prompts(num_prompts: int, seed: int = 42) -> List[str]:
    random.seed(seed)
    prompts = set()

    # 40% music, 30% sound effects, 20% ambient, 10% complex
    categories = [
        (generate_music_prompt, 0.4),
        (generate_sound_effect_prompt, 0.3),
        (generate_ambient_prompt, 0.2),
        (generate_complex_prompt, 0.1)
    ]

    while len(prompts) < num_prompts:
        rand = random.random()
        cumulative = 0
        for generator, weight in categories:
            cumulative += weight
            if rand < cumulative:
                prompt = generator()
                prompts.add(prompt)
                break

    return list(prompts)


def save_prompts(prompts: List[str], output_file: str):
    with open(output_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    print(f"Saved {len(prompts)} prompts to {output_file}")


if __name__ == "__main__":
    num_prompts = 10_000

    print(f"Generating {num_prompts} diverse audio prompts...")
    prompts = generate_prompts(num_prompts)

    print("\nExample prompts:")
    for i in range(10):
        print(f"  {i+1}. {prompts[i]}")

    save_prompts(prompts, "prompts.json")

    print(f"\nTotal unique prompts: {len(prompts)}")
