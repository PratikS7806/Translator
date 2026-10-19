def shrink_file(filepath, num_lines=100000):
    # Read only the first 100,000 lines
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [next(f) for _ in range(num_lines)]
        
    # Overwrite the massive file with just the sliced data
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"Successfully shrunk {filepath} to {num_lines} lines.")

try:
    shrink_file("data/multi30k/train.de", 100000)
    shrink_file("data/multi30k/train.en", 100000)
    print("Data slice complete! Ready to restart.")
except StopIteration:
    print("Files already have fewer than 100,000 lines.")
    