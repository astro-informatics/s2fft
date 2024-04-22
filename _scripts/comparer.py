import re

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

def compare_files(file1, file2):
    numbers1 = read_complex_numbers(file1)
    numbers2 = read_complex_numbers(file2)

    max_abs_error = 0
    for i in range(min(len(numbers1), len(numbers2))):
        abs_error = abs(numbers1[i] - numbers2[i])
        if abs_error > max_abs_error:
            max_abs_error = abs_error
    
    return max_abs_error

file1 = 'fft_noshift.log'  # Path to the first file
file2 = 'actual.log'  # Path to the second file#
file3 = 'fft_shifted.log'  # Path to the second file
file4 = 'actualshifted.log'  # Path to the second file

numbers1 = read_complex_numbers(file1)
numbers2 = read_complex_numbers(file2)

print(numbers2[0])
print(numbers1[0])

max_abs_error = compare_files(file1, file2)
print("Max absolute error without:", max_abs_error)
max_abs_error = compare_files(file3, file4)
print("Max absolute error with shift:", max_abs_error)
