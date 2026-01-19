import json
try:
    with open('smart_home_corpus.json', 'r', encoding='utf-8') as f:
        old_data = json.load(f)
except FileNotFoundError:
    print("ERROR: 'smart_home_corpus.json' not found. Please make sure it is in the correct directory.")
    exit()

# INTENT MAPPING (We group similar commands)
intent_map = {
    "LIGHT_ON": [1, 2, 12, 13], # Turn on the light, turn on the lamp
    "LIGHT_OFF": [3, 4, 14, 15],# Turn off the light, turn off, turn off the lamp, turn off
    "LIGHT_UP": [5, 8, 10],      # Increase light, increase brightness, increase illumination
    "LIGHT_DOWN": [6, 7, 9, 11], # Dim the light, reduce it, reduce brightness
    "COLOR_RED": [16],
    "COLOR_BLUE": [17],
    "COLOR_GREEN": [18],
    "PLUG_ON": [19],
    "PLUG_OFF": [20],
    "AC_ON": [21, 23, 25, 35],   #Air conditioning,heating, combi boiler ON
    "AC_OFF": [22, 24, 26, 36],  #Air conditioning,heating, combi boiler OFF
    "SET_HEAT": [27, 29, 31, 33], # Heat, increase the temperature, heat the house
    "SET_COOL": [28, 30, 32, 34], # Cool, lower the temperature, cool the house
    "GET_TEMP": [37, 38, 39],    # What is the temperature in the room, what is the temperature?
    "CHANGE_MODE": [40],
    "FAN_ON": [41],
    "FAN_OFF": [42],
    "FAN_UP": [43],
    "FAN_DOWN": [44],
    "TV_ON": [45, 47],
    "TV_OFF": [46, 48],
    "MEDIA_ON": [49, 51],        # Multimedia, Music ON
    "MEDIA_OFF": [50, 52],       # Multimedia, Music OFF
    "BLIND_OPEN": [53, 55],      # OPEN Blinds, Curtains
    "BLIND_CLOSE": [54, 56],     # Shutter, Curtain CLOSE
    "SECURITY_ON": [57],
    "CONFIRM_YES": [58],
    "CONFIRM_NO": [59],
    "SCENE_PARTY": [60],
    "SCENE_RELAX": [61],
    "SCENE_SLEEP": [62],
    "SCENE_HOME": [63],
    "SCENE_AWAY": [64],
    "SCENE_MORNING": [65],
    "SCENE_NIGHT": [66],
    "SCENE_MOVIE": [67],
    "SCENE_WORK": [68],
    "SCENE_SPORT": [69, 70],
    "SCENE_AUTO": [71]
}

new_corpus = []
id_counter = 1

print(f"Processing a total set of {len(old_data)} old commands...")

for intent_name, old_ids in intent_map.items():
    merged_variations = []
    
    # Collect variations of related IDs
    for item in old_data:
        if item['id'] in old_ids:
            merged_variations.extend(item['variations'])
    
    # Clear (set) duplicates and turn them into a list
    unique_variations = list(set(merged_variations))
    
    if unique_variations:
        new_corpus.append({
            "id": id_counter,
            "intent": intent_name,
            # We now use the name intent as the target command
            "target_command": intent_name, 
            "original_commands": old_ids,
            "variations": unique_variations
        })
        id_counter += 1

# Save file
output_file = 'smart_home_corpus.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(new_corpus, f, ensure_ascii=False, indent=2)

print("="*60)
print(f"✓ OPTIMIZATION COMPLETED!")
print(f"✓ {len(old_data)} old command -> {len(new_corpus)} converted to new INTENT group.")
print(f"✓ New file: '{output_file}'")
print("="*60)
print("\nWHAT YOU NEED TO DO NOW:")
print(" Now you can run phase_a_train_model.py")