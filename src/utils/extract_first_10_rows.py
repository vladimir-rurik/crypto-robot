import csv
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_first_10_rows.py <input_csv>")
        sys.exit(1)
    
    original_file = sys.argv[1]
    
    # Validate that the file exists
    if not os.path.isfile(original_file):
        print(f"Error: File '{original_file}' not found.")
        sys.exit(1)
    
    # Construct output file name
    base_name, extension = os.path.splitext(original_file)
    new_file = base_name + "_10_first_rows" + extension
    
    try:
        with open(original_file, 'r', newline='', encoding='utf-8') as infile, \
             open(new_file, 'w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            # Read and write the header
            try:
                header = next(reader)
                writer.writerow(header)
            except StopIteration:
                print("Input CSV is empty. No rows to write.")
                return
            
            # Write up to 10 rows
            for i, row in enumerate(reader, start=1):
                if i <= 10:
                    writer.writerow(row)
                else:
                    break
        
        print(f"Successfully wrote the first 10 rows of '{original_file}' to '{new_file}'")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
