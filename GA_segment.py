import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import concurrent.futures
import threading
import time

class GA_Segment:
    def __init__(self, input: np.ndarray, image_path="", n_population=20, n_iterations=100, n_bins=256,
                 n_thresholds=5, p_selection=0.1, p_crossover=0.8, p_mutation=0.1):
        assert sum([p_selection, p_crossover, p_mutation]
                   ) == 1, 'Total sum of proportions have to be 1!'
        if image_path == "":
            self.image = input
        else:
            self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.image = self.median_filter(self.image)
        self.n_population = n_population
        self.n_iterations = n_iterations
        self.n_bins = n_bins
        self.n_thresholds = n_thresholds
        self.p_selection = p_selection
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.im_mask = np.zeros_like(self.image, dtype=np.float32)
        self.im_seg = np.zeros_like(self.image, dtype=np.float32)
        self.population = self.initialization()

    def median_filter(self, image):
        """Áp dụng bộ lọc trung vị để giảm nhiễu."""
        return cv2.medianBlur(image, 3)

    def initialization(self):
        """Khởi tạo quần thể."""
        return np.round(np.random.uniform(0, 1, (self.n_population, int(np.ceil(np.log2(self.n_bins)) * self.n_thresholds))))

    def fitness(self, population):
        """Tính toán giá trị fitness cho từng cá thể."""
        ranking = []
        thresholds = self.convert_thresholds(population)
        image_vec = self.image.flatten()

        for threshold in thresholds:
            ranking.append(self.fitness_one(image_vec, threshold))

        return np.array(ranking)

    def convert_thresholds(self, population):
        """Chuyển đổi thresholds từ dạng nhị phân sang thập phân."""
        return np.array([self.threshold_bin2dec(ind) for ind in population])

    def fitness_one(self, image_vec, thresholds_vec):
        """Tính toán giá trị fitness cho một cá thể."""
        ranking = 1
        thresholds_vec = np.sort(thresholds_vec)

        split_points = np.concatenate(
            ([0], thresholds_vec, [np.max(image_vec)]))

        for i in range(len(split_points) - 1):
            left, right = split_points[i], split_points[i + 1]
            mask = (image_vec >= left) & (image_vec < right)
            segment = image_vec[mask]
            variance = np.var(segment) if len(segment) > 0 else 1
            ranking += variance

        return ranking

    def first_best(self, ranking, population, p_selection, new_population):
        """Chọn lọc các cá thể tốt nhất."""
        population_size = len(population)
        best_indices = np.argsort(ranking)
        for i in range(round(p_selection * population_size)):
            new_population.append(population[best_indices[i]])
        return new_population

    def crossover(self, population, p_crossover, new_population):
        """Thực hiện lai ghép giữa các cá thể."""
        population_size = len(population)
        parent_first = np.random.permutation(population_size)
        parent_second = np.random.permutation(population_size)
        n_crossovers = round(p_crossover * population_size) // 2

        for i in range(n_crossovers):
            desc_first, desc_second = self.crossover_one(
                population[parent_first[i]], population[parent_second[i]])
            new_population.append(desc_first)
            new_population.append(desc_second)

        return new_population

    def crossover_one(self, parent_first, parent_second):
        """Thực hiện lai ghép một cặp cha mẹ."""
        parent_size = len(parent_first)
        point = np.random.randint(1, parent_size)
        desc_first = np.concatenate(
            (parent_first[:point], parent_second[point:]))
        desc_second = np.concatenate(
            (parent_second[:point], parent_first[point:]))
        return desc_first, desc_second

    def mutation(self, population, p_mutation, new_population):
        """Thực hiện đột biến trên một số cá thể."""
        population_size = len(population)
        mutation_order = np.random.permutation(population_size)

        for i in range(round(p_mutation * population_size)):
            new_population.append(self.mutate_one(
                population[mutation_order[i]]))

        return new_population

    def mutate_one(self, chromosome):
        """Đột biến một gene trong nhiễm sắc thể."""
        new_chromosome = chromosome.copy()
        gene = np.random.randint(len(chromosome))
        new_chromosome[gene] = 1 - new_chromosome[gene]  # Đảo bit 0 <-> 1
        return new_chromosome

    def threshold_bin2dec(self, bin_thresholds):
        """Chuyển đổi từ nhị phân sang thập phân."""
        threshold_length = len(bin_thresholds) // self.n_thresholds
        dec_thresholds = [int("".join(map(lambda x: str(int(x)), bin_thresholds[i:i+threshold_length])), 2)
                          for i in range(0, len(bin_thresholds), threshold_length)]
        return np.array(dec_thresholds)

    def accept_solution(self, population):
        """Chấp nhận nghiệm tối ưu sau quá trình tiến hóa."""
        self.im_seg = np.zeros_like(self.image, dtype=np.float32)
        segmentation_value = 1 / self.n_thresholds

        ranking = self.fitness(population)
        best_genome = population[np.argmin(ranking)]
        thresholds = np.sort(self.threshold_bin2dec(best_genome))

        value = 0
        split_points = np.concatenate(([0], thresholds, [np.max(self.image)]))

        for i in range(len(split_points) - 1):
            left, right = split_points[i], split_points[i + 1]
            mask = (self.image >= left) & (self.image < right)
            self.im_seg[mask] = value
            value += segmentation_value

        self.im_mask = np.where((self.im_seg >= 0.6) & (
            self.im_seg <= 1), 1, 0).astype(np.uint8) * 255

    def print(self):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(self.image, cmap='gray')
        plt.title('Original Image')

        plt.subplot(1, 3, 2)
        plt.imshow(self.im_seg, cmap='gray')
        plt.title('Segmented Image')

        hist, bins = np.histogram(self.im_seg.ravel(), bins=10, range=[0, 1])
        print("Histogram of Segmentation:")
        for b, h in zip(bins[:-1], hist):
            print(f"Bin {b:.2f} - {b+0.1:.2f}: {h}")

    def run(self):
        for _ in range(self.n_iterations):
            new_population = []

            ranking = self.fitness(self.population)

            new_population = self.first_best(
                ranking, self.population, self.p_selection, new_population)
            new_population = self.crossover(
                self.population, self.p_crossover, new_population)
            new_population = self.mutation(
                self.population, self.p_mutation, new_population)

            self.population = np.array(new_population)

        self.accept_solution(self.population)


def GA_Segment_nii(data_in):
    """
    Hàm segment ảnh nii

    Đầu vào:
        data_in: Ảnh nii dưới dạng numpy array (128x128x128)

    Đầu ra:
        data_out: Ảnh đã segment (128x128x128)
    """

    # Chuẩn hóa dữ liệu về khoảng 0 - 255 để xử lý
    norm_data = cv2.normalize(data_in, None, 0, 255, cv2.NORM_MINMAX)
    norm_data = np.uint8(norm_data)  # Chuyển về uint8

    # Tạo mảng đầu ra để lưu kết quả segmentation
    data_out = np.zeros_like(norm_data, dtype=np.uint8)

    processed_slices = 0
    total_slices = norm_data.shape[2]
    def print_status():
        while processed_slices < total_slices:
            print(f"Đã xử lý {processed_slices}/{total_slices} lát cắt...")
            time.sleep(10)
    
    # Xử lý từng lát cắt 2D trong ảnh 3D
    # Định nghĩa hàm xử lý từng lát cắt
    def process_slice(i):
        nonlocal processed_slices
        slice_img = norm_data[:, :, i]
        ga_segment = GA_Segment(slice_img)
        ga_segment.run()
        data_out[:, :, i] = ga_segment.im_mask
        processed_slices += 1
    # Chạy luồng in trạng thái
    status_thread = threading.Thread(target=print_status, daemon=True)
    status_thread.start()
    # Dùng ThreadPoolExecutor để chạy đa luồng
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_slice, range(total_slices))

    return data_out


# Đọc file
file_path = "Brats18_CBICA_AAM_1_flair.nii"
nii_img = nib.load(file_path)

# Lấy dữ liệu ảnh dưới dạng numpy array
nii_data = nii_img.get_fdata()

# sử dụng hàm
seg_data = GA_Segment_nii(nii_data)


print("Kích thước đầu vào:", nii_data.shape)
print("Kích thước đầu ra:", seg_data.shape)
