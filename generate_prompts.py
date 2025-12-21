import json
import random
from typing import List


MUSIC_GENRES = [
    "electronic", "jazz", "classical", "rock", "hip hop", "ambient", "techno",
    "house", "trance", "dubstep", "lo-fi", "folk", "country", "blues", "reggae",
    "metal", "punk", "indie", "pop", "r&b", "soul", "funk", "disco", "trap",
    "drum and bass", "synthwave", "vaporwave", "garage", "breakbeat", "IDM",
    "downtempo", "chillwave", "future bass", "hardstyle", "psytrance", "minimal",
    "deep house", "progressive", "electro", "dancehall", "afrobeat", "bossa nova",
    "swing", "bebop", "bluegrass", "grunge", "shoegaze", "post-rock", "math rock"
]

MUSIC_MOODS = [
    "upbeat", "melancholic", "energetic", "calm", "dark", "bright", "dreamy",
    "aggressive", "peaceful", "nostalgic", "mysterious", "joyful", "sad", "intense",
    "relaxing", "euphoric", "eerie", "somber", "playful", "romantic", "dramatic",
    "ethereal", "groovy", "hypnotic", "triumphant", "ominous", "serene", "chaotic",
    "whimsical", "contemplative", "inspiring", "haunting", "cheerful", "tender",
    "powerful", "delicate", "atmospheric", "cinematic", "mellow", "vibrant"
]

MUSIC_INSTRUMENTS = [
    "piano", "guitar", "drums", "bass", "violin", "saxophone", "synthesizer",
    "flute", "trumpet", "cello", "harmonica", "accordion", "harp", "organ",
    "clarinet", "trombone", "banjo", "mandolin", "ukulele", "sitar", "tabla",
    "didgeridoo", "bagpipes", "xylophone", "marimba", "vibraphone", "oboe",
    "bassoon", "tuba", "french horn", "electric guitar", "acoustic guitar",
    "steel drums", "bongo drums", "congas", "timpani", "glockenspiel", "kalimba",
    "theremin", "vocoder", "mellotron", "music box"
]

MUSIC_SETTINGS = [
    "in a concert hall", "in a small cafe", "in a studio", "at a festival",
    "in a church", "in a jazz club", "in a bedroom", "on a street corner",
    "in a cathedral", "at a wedding", "in a nightclub", "at a rave",
    "in an underground venue", "at a beach party", "in a warehouse",
    "at an outdoor amphitheater", "in a living room", "in a barn",
    "at a rooftop bar", "in a subway station", "at a campfire", "in a recording booth"
]

ANIMALS = [
    "dog barking", "cat meowing", "bird chirping", "owl hooting", "wolf howling",
    "horse neighing", "cow mooing", "rooster crowing", "cricket chirping",
    "frog croaking", "lion roaring", "elephant trumpeting", "dolphin clicking",
    "monkey screeching", "bear growling", "snake hissing", "eagle screeching",
    "duck quacking", "goose honking", "pig grunting", "sheep bleating",
    "donkey braying", "turkey gobbling", "peacock calling", "parrot squawking",
    "seagull crying", "crow cawing", "bee buzzing", "mosquito buzzing",
    "cicada singing", "whale singing", "seal barking", "penguin calling",
    "bat screeching", "coyote howling", "hyena laughing", "tiger roaring"
]

NATURE_SOUNDS = [
    "rain falling", "thunder rumbling", "wind blowing", "waves crashing",
    "waterfall flowing", "stream babbling", "leaves rustling", "fire crackling",
    "snow crunching", "ice cracking", "river flowing", "ocean waves",
    "raindrops on water", "hail falling", "blizzard howling", "tornado wind",
    "avalanche rumbling", "earthquake shaking", "volcano erupting", "lightning striking",
    "rain on leaves", "wind through trees", "bamboo forest", "rustling grass",
    "desert wind", "cave dripping", "geyser erupting", "mud bubbling",
    "sandstorm", "forest ambience", "jungle sounds", "swamp noises"
]

URBAN_SOUNDS = [
    "car engine", "traffic noise", "train passing", "airplane flying",
    "construction work", "crowd chatter", "footsteps", "door closing",
    "phone ringing", "keyboard typing", "glass breaking", "siren wailing",
    "subway train", "bus brakes", "motorcycle revving", "car horn honking",
    "bicycle bell", "skateboard rolling", "helicopter overhead", "jet engine",
    "jackhammer", "drill", "saw cutting", "hammer pounding",
    "street musician", "market vendors", "children playing", "people talking",
    "elevator ding", "escalator", "automatic doors", "shopping cart",
    "cash register", "printer", "copier", "fax machine",
    "ambulance siren", "police siren", "fire truck", "car alarm"
]

HOUSEHOLD_SOUNDS = [
    "washing machine running", "vacuum cleaner", "microwave beeping",
    "clock ticking", "water dripping", "doorbell ringing", "chair scraping",
    "dishes clattering", "keys jingling", "refrigerator humming", "fan blowing",
    "air conditioner", "heater running", "toilet flushing", "shower running",
    "bath filling", "sink draining", "tap water", "boiling water",
    "blender", "coffee maker", "toaster popping", "kettle whistling",
    "oven timer", "dishwasher", "dryer spinning", "iron steaming",
    "door creaking", "floorboards creaking", "stairs creaking", "window closing",
    "blinds rattling", "curtains swooshing", "light switch", "pen clicking"
]

# Ambient environments
ENVIRONMENTS = [
    "forest", "beach", "city street", "mountain", "desert", "jungle",
    "park", "library", "restaurant", "gym", "office", "subway station",
    "airport", "hospital", "school", "market", "canyon", "valley",
    "lake", "river", "ocean", "meadow", "prairie", "swamp", "marsh",
    "cave", "tunnel", "bridge", "harbor", "dock", "pier", "marina",
    "farm", "countryside", "village", "town square", "alleyway", "rooftop",
    "basement", "attic", "garage", "warehouse", "factory", "mall",
    "theater", "cinema", "museum", "gallery", "stadium", "arena",
    "temple", "monastery", "cathedral", "chapel", "mosque", "synagogue"
]

WEATHER = [
    "rainy day", "sunny afternoon", "stormy night", "windy morning",
    "foggy evening", "snowy day", "thunderstorm", "light drizzle",
    "heavy downpour", "cloudy day", "clear sky", "overcast weather",
    "hailstorm", "blizzard", "heat wave", "cold front", "misty morning",
    "humid day", "dry season", "monsoon", "hurricane", "typhoon"
]

TIMES = [
    "morning", "afternoon", "evening", "night", "dawn", "dusk", "midnight",
    "sunrise", "sunset", "noon", "twilight", "late night", "early morning",
    "mid-afternoon", "late evening", "pre-dawn", "witching hour"
]

# Add descriptive adjectives
ADJECTIVES = [
    "quiet", "loud", "busy", "peaceful", "chaotic", "serene", "hectic",
    "tranquil", "bustling", "empty", "crowded", "isolated", "remote",
    "urban", "rural", "industrial", "natural", "synthetic", "organic",
    "rhythmic", "melodic", "dissonant", "harmonic", "percussive"
]


def generate_music_prompt() -> str:
    templates = [
        f"{random.choice(MUSIC_MOODS)} {random.choice(MUSIC_GENRES)} music",
        f"{random.choice(MUSIC_INSTRUMENTS)} playing {random.choice(MUSIC_GENRES)} music",
        f"{random.choice(MUSIC_GENRES)} {random.choice(MUSIC_INSTRUMENTS)} {random.choice(MUSIC_SETTINGS)}",
        f"{random.choice(MUSIC_MOODS)} {random.choice(MUSIC_INSTRUMENTS)} melody",
        f"{random.choice(MUSIC_GENRES)} track with {random.choice(MUSIC_INSTRUMENTS)}",
        f"{random.choice(ADJECTIVES)} {random.choice(MUSIC_GENRES)} music",
        f"{random.choice(MUSIC_INSTRUMENTS)} and {random.choice(MUSIC_INSTRUMENTS)} duet",
        f"{random.choice(MUSIC_MOODS)} {random.choice(MUSIC_GENRES)} beat",
        f"solo {random.choice(MUSIC_INSTRUMENTS)} performance",
        f"{random.choice(MUSIC_GENRES)} music with {random.choice(MUSIC_MOODS)} atmosphere",
        f"{random.choice(MUSIC_INSTRUMENTS)} improvisation in {random.choice(MUSIC_GENRES)} style",
        f"fast-paced {random.choice(MUSIC_GENRES)} rhythm",
        f"slow {random.choice(MUSIC_INSTRUMENTS)} ballad",
        f"{random.choice(MUSIC_GENRES)} fusion with {random.choice(MUSIC_GENRES)}",
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
        f"{sound} in the distance",
        f"{sound} close by",
        f"{sound} echoing",
        f"{sound} constantly",
        f"{sound} intermittently",
        f"sharp {sound}",
        f"muffled {sound}",
        f"clear {sound}",
        f"faint {sound}",
    ]
    return random.choice(templates)


def generate_ambient_prompt() -> str:
    templates = [
        f"ambient sounds of a {random.choice(ENVIRONMENTS)}",
        f"{random.choice(ENVIRONMENTS)} on a {random.choice(WEATHER)}",
        f"{random.choice(ENVIRONMENTS)} at {random.choice(TIMES)}",
        f"{random.choice(WEATHER)} in the {random.choice(ENVIRONMENTS)}",
        f"peaceful {random.choice(TIMES)} in a {random.choice(ENVIRONMENTS)}",
        f"{random.choice(ADJECTIVES)} {random.choice(ENVIRONMENTS)} atmosphere",
        f"{random.choice(ENVIRONMENTS)} soundscape",
        f"{random.choice(ADJECTIVES)} {random.choice(ENVIRONMENTS)} at {random.choice(TIMES)}",
        f"nature sounds from a {random.choice(ENVIRONMENTS)}",
        f"{random.choice(ENVIRONMENTS)} during {random.choice(WEATHER)}",
        f"{random.choice(TIMES)} ambience in a {random.choice(ENVIRONMENTS)}",
        f"{random.choice(WEATHER)} atmosphere",
    ]
    return random.choice(templates)


def generate_complex_prompt() -> str:
    templates = [
        f"{random.choice(ANIMALS)} with {random.choice(NATURE_SOUNDS)} in the background",
        f"{random.choice(MUSIC_INSTRUMENTS)} playing while {random.choice(URBAN_SOUNDS)}",
        f"{random.choice(NATURE_SOUNDS)} and {random.choice(ANIMALS)} in a {random.choice(ENVIRONMENTS)}",
        f"{random.choice(ANIMALS)} and {random.choice(ANIMALS)} together",
        f"{random.choice(NATURE_SOUNDS)} mixed with {random.choice(URBAN_SOUNDS)}",
        f"{random.choice(MUSIC_INSTRUMENTS)} accompanied by {random.choice(NATURE_SOUNDS)}",
        f"{random.choice(URBAN_SOUNDS)} in a {random.choice(ENVIRONMENTS)} during {random.choice(WEATHER)}",
        f"{random.choice(HOUSEHOLD_SOUNDS)} with {random.choice(ANIMALS)} nearby",
        f"{random.choice(MUSIC_MOODS)} {random.choice(MUSIC_GENRES)} with {random.choice(NATURE_SOUNDS)}",
        f"{random.choice(ANIMALS)} over {random.choice(MUSIC_GENRES)} music",
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

    # Safety mechanism: give up if we can't generate enough unique prompts
    max_attempts = num_prompts * 10
    attempts = 0
    last_size = 0
    stall_count = 0

    while len(prompts) < num_prompts and attempts < max_attempts:
        attempts += 1
        rand = random.random()
        cumulative = 0
        for generator, weight in categories:
            cumulative += weight
            if rand < cumulative:
                prompt = generator()
                prompts.add(prompt)
                break

        if attempts % 100 == 0:
            if len(prompts) == last_size:
                stall_count += 1
                if stall_count > 10:
                    print(f"Warning: Stalled at {len(prompts)} unique prompts")
                    break
            else:
                stall_count = 0
            last_size = len(prompts)

    if len(prompts) < num_prompts:
        print(f"Warning: Could only generate {len(prompts)} unique prompts out of {num_prompts} requested")

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
