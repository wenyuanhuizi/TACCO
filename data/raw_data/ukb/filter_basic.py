def preprocess_data(input_filename, output_filename):
    # Define the structure
    keep_indices = [
        (0, 1140),      # Keep first 1140 lines
        (1140, 1304),   # Keep next 164 lines (1140 + 164)
        (1468, 1549),   # Skip 164 lines, keep next 81 lines (1304 + 164 + 81)
        (1624, 1868),   # Skip 81 lines, keep next 244 lines (1549 + 81 + 244)
    ]

    # Read the dataset
    with open(input_filename, 'r') as file:
        lines = file.readlines()

    # Select the specified lines based on the structure
    selected_lines = []
    for start, end in keep_indices:
        selected_lines.extend(lines[start:end])

    # Write the selected lines to the new file
    with open(output_filename, 'w') as file:
        file.writelines(selected_lines)

    print(f"Selected lines written to {output_filename}")

    # Count the number of unique medical codes in the filtered data
    unique_codes = set()
    for line in selected_lines:
        codes = map(int, line.strip().split(','))
        unique_codes.update(codes)

    # Print the number of unique medical codes
    print(f"Number of unique medical codes in the filtered data: {len(unique_codes)}")

# File names
input_filename = '/Users/wenyuanhuizi/Desktop/TACCO/data/raw_data/ukb/hyperedges-ukb.txt'
output_filename = '/Users/wenyuanhuizi/Desktop/TACCO/data/raw_data/ukb/hyperedges_basic_ukbran.txt'

# Preprocess the data
preprocess_data(input_filename, output_filename)
