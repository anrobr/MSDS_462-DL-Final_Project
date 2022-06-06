def read_in_class_labels(dir, file_name):    
    class_labels = []
    with open(dir.joinpath(file_name), 'r') as file:
        while (line := file.readline().rstrip()):
            class_labels.append(line)
    return class_labels