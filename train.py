from math import sqrt
from typing import Callable

# defining image return type
Pixel = int
Image = list[Pixel]
Images = list[Image]

# defining label return type
Label = int
Labels = list[Label]

def read_file(filename: str, amount: int | None = None) -> Images | Labels:
    to_int = lambda b: int.from_bytes(b)

    with open(filename, 'rb') as f:
        if 'images' in filename:
            _, items, rows, cols = [to_int(f.read(4)) for _ in range(4)]

            return [
                [px for px in f.read(rows * cols)]
                for _ in range(items if not amount else amount)
            ]
        
        elif 'labels' in filename:
            _, items = [to_int(f.read(4)) for _ in range(2)]
            
            return [
                label
                for label in f.read(items if not amount else amount)
            ]
    
    raise Exception(f'Unable to process file {filename}')

def get_files(amount: int = 500) -> list[Images | Labels]:
    return [
        read_file(f'./data/{name}', amount) 
        for name in (
            'train-images-idx3-ubyte',
            'train-labels-idx1-ubyte',
            't10k-images-idx3-ubyte',
            't10k-labels-idx1-ubyte',
        )
    ]

# defining metrics to see which one provides best performance
def euclidean_metric(A: Image, B: Image) -> float:
    return sqrt(sum(
        (a - b) ** 2
        for (a, b) in zip(A, B)
    ))

def manhattan_metric(A: Image, B: Image) -> float:
    return sum(abs(a - b) for (a, b) in zip(A, B))

def kNN(training_images: Images, 
        training_labels: Labels, 
        test_image: Image, 
        dist_metric: Callable[[Image, Image], float] = euclidean_metric, 
        k=3) -> list[Label]:
    
    distances: list[tuple[Label, float]] = [
        (label, dist_metric(training_image, test_image))
        for (label, training_image) in zip(training_labels, training_images) 
    ]

    return [
        data[0] 
        for data in sorted(distances, key=lambda t: t[1])[:k]
    ]
