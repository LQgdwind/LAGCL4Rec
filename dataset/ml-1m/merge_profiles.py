import json
import os

# File paths
BASE_DIR = "./dataset/ml-1m"
USER_PROFILES_PATH = "./dataset/ml-1m/user_profiles.json"
ITEM_PROFILES_PATH = "./dataset/ml-1m/item_profiles.json"
TRAIN_PATH = "./dataset/ml-1m/train.txt"
OUTPUT_PATH = "./dataset/ml-1m/merged_profiles.json"

def load_json_array(file_path, id_key):
    """Load a JSON array file into a dictionary indexed by id"""
    data = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Fix potentially malformed JSON by ensuring it's a valid array
            if not content.strip().startswith('['):
                content = '[' + content
            if not content.strip().endswith(']'):
                content = content + ']'
            
            json_array = json.loads(content)
            
            for item in json_array:
                if id_key in item:
                    data[item[id_key]] = item
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        # Try alternate approach with line-by-line parsing
        try:
            data = {}
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                start_idx = content.find('{')
                if start_idx != -1:
                    content = content[start_idx:]
                end_idx = content.rfind('}')
                if end_idx != -1:
                    content = content[:end_idx+1]
                
                parts = content.split('},{')
                for i, part in enumerate(parts):
                    if i == 0 and not part.startswith('{'):
                        part = '{' + part
                    if i == len(parts) - 1 and not part.endswith('}'):
                        part = part + '}'
                    if i > 0 and i < len(parts) - 1:
                        part = '{' + part + '}'
                    
                    try:
                        item = json.loads(part)
                        if id_key in item:
                            data[item[id_key]] = item
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Alternate parsing approach failed: {e}")
    
    return data

def parse_interactions(file_path):
    """Parse interaction data from txt file and group by user"""
    user_interactions = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                user_id, item_id, interaction_type = parts[0], parts[1], parts[2]
                
                if user_id not in user_interactions:
                    user_interactions[user_id] = {'positive': [], 'negative': []}

                if interaction_type == '1':
                    user_interactions[user_id]['positive'].append(item_id)
                elif interaction_type == '-1':
                    user_interactions[user_id]['negative'].append(item_id)
    
    return user_interactions

def merge_data():
    """Merge user profiles, item profiles and interactions into a single JSON file"""
    try:
        print("Loading user profiles...")
        user_profiles = load_json_array(USER_PROFILES_PATH, 'user_id')
        print(f"Loaded {len(user_profiles)} user profiles")
        
        print("Loading item profiles...")
        item_profiles = load_json_array(ITEM_PROFILES_PATH, 'movie_id')
        print(f"Loaded {len(item_profiles)} item profiles")
        
        print("Loading interaction data...")
        user_interactions = parse_interactions(TRAIN_PATH)
        print(f"Loaded interactions for {len(user_interactions)} users")
        
        merged_data = []
        
        print("Merging data...")
        processed_count = 0
        for user_id, interactions in user_interactions.items():
            user_entry = {
                "user_id": user_id,
                "user_profile": user_profiles.get(user_id, {})
            }
        
            positive_items = []
            for item_id in interactions['positive']:
                if item_id in item_profiles:
                    positive_items.append({
                        "item_id": item_id,
                        "item_profile": item_profiles[item_id]
                    })
            user_entry["positive_items"] = positive_items
            
            negative_items = []
            for item_id in interactions['negative']:
                if item_id in item_profiles:
                    negative_items.append({
                        "item_id": item_id,
                        "item_profile": item_profiles[item_id]
                    })
            user_entry["negative_items"] = negative_items
            
            merged_data.append(user_entry)
            
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} users")
        
        print(f"Writing merged data for {len(merged_data)} users to output file...")

        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
        print(f"Merged data successfully written to {OUTPUT_PATH}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    merge_data() 