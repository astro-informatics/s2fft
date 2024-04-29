import re
import argparse

def read_complex_numbers(file_path):
    numbers = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r'\[(\d+)\]\s*(\()?(.*)\s*\+\s*(.*)(j|i).*', line)
            if match:
                groups = match.groups()
                real_part = float(groups[2])
                imag_str = groups[3].replace('j', '').replace('i', '')
                imag_part = float(imag_str)
                numbers.append(complex(real_part, imag_part))
    return numbers

def compare_files(file1, file2, print_errors=False):
    numbers1 = read_complex_numbers(file1)
    numbers2 = read_complex_numbers(file2)

    max_abs_error = 0
    for i in range(min(len(numbers1), len(numbers2))):
        #print(f"numbers1[{i}]: {numbers1[i]} numbers2[{i}]: {numbers2[i]}")
        abs_error = abs(numbers1[i] - numbers2[i])
        if abs_error > max_abs_error:
            max_abs_error = abs_error
        if abs_error > 1 and print_errors:
            print(f"Error at index {i}: {abs_error}")
    
    return max_abs_error

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Comparer')
    parser.add_argument('-f' , '--files' , nargs=2, help='Files to compare',required=True)
    parser.add_argument('-p' , '--print', help='Print errors', action='store_true')

    args = parser.parse_args()

    max_abs_error = compare_files(args.files[0] , args.files[1], args.print)
    print("Max absolute error without:", max_abs_error)
        
