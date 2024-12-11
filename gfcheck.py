import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



def trim_to_min_rows(data1, data2):
    """裁剪两个数组，使其行数相同"""
    min_rows = min(len(data1), len(data2))
    return data1[:min_rows], data2[:min_rows]



def check_convergence(case, type, columns1, columns2, tolerance_percentage, consecutive_points, output_image_name):
    """
    Determines whether the data in the specified columns of two txt files are close enough and plots a line graph.
    
    Parameters.
    file1, file2: str, paths of the two txt files.
    columns: list of int, index of columns to be compared (counting from 1)
    tolerance_percentage: float, tolerance threshold, percentage of larger value
    consecutive_points: int, how many consecutive points the difference needs to be less than the threshold value
    output_image_name: str, path to save the folded image to
    """
    if type == 'up':
        file1 = case + '.gc1'
    elif type == 'down':
        file1 = case + '.gc2'
    
    file2 =  Path.cwd() / 'imp.0' / 'Gf.out.50.1'


    # read file
    data1 = np.loadtxt(file1, usecols=[col - 1 for col in columns1]) 
    data2 = np.loadtxt(file2, usecols=[col - 1 for col in columns2])
    data1, data2 = trim_to_min_rows(data1, data2)
    # data1 = data1[:100, :]
    # data2 = data2[:100, :]

    # total data number check
    if data1.shape[0] != data2.shape[0]:
        raise ValueError("file rows number is different")
    
    # results
    is_converged = []
    
    max_value = max(np.max(np.abs(data1)), np.max(np.abs(data2)))

    # check
    for i, col in enumerate(columns1):
        diff = np.abs(data1[:, i] - data2[:, i])  
        #max_values = np.maximum(np.abs(data1[:, i]), np.abs(data2[:, i]))
 
        threshold = tolerance_percentage / 100.0 * max_value  
        within_tolerance = diff < threshold  
        
        # check the final points
        if np.all(within_tolerance[-consecutive_points:]):
            is_converged.append(True)
        else:
            is_converged.append(False)

        # figure
        plt.plot(data1[:, i], label=f'gc1 - Column {col}')
        plt.plot(data2[:, i], label=f'GF.out - Column {col}')
    
    # figure caption
    plt.xlabel('Row Number')
    plt.ylabel('Value')
    plt.legend()
    
    # get figure
    plt.savefig(output_image_name)
    plt.close()
    
    # output
    if all(is_converged):
        return True
    else:
        return False


def main():

    GF_up = check_convergence('RuO2', 'up', [3,5,7,9,11], [3,5,7,9,11], 5, 50, 'GF_up.png')
    GF_down = check_convergence('RuO2', 'down', [3,5,7,9,11], [13,15,17,19,21], 5, 50, 'GF_down.png')
    print(GF_up, GF_down)


if __name__ == '__main__':  
    main()