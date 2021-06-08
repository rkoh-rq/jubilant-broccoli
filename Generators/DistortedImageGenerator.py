import numpy as np
import cv2
from torch import from_numpy

class DistortedImageGenerator:
    def __init__(self, 
                module_generator, 
                edge_tolerance=2, 
                use_pre_generated_maps=False, 
                num_maps_to_generate=1000,
                d_K0 = 0.0000003,
                d_xc = 5,
                d_yc = 5,
                d_alpha = 0.001,
                d_beta = 0.001,
                d_gamma = 0.001,
                d_sx = 0.1,
                d_sy = 0.1,
                d_cx = 20,
                d_cy = 20):
        self.module_generator = module_generator
        self.height, self.width, _ = module_generator.shape_in_px
        self.edge_tolerance = edge_tolerance  # How much of the edge can go missing

        self.cam_matrix = np.identity(3, dtype=np.float32)
        self.cam_matrix[0][2] = self.width//2
        self.cam_matrix[1][2] = self.height//2

        self.use_pre_generated_maps = use_pre_generated_maps
        self.num_maps_to_generate = num_maps_to_generate

        if self.use_pre_generated_maps:
            self.maps_x = []
            self.maps_y = []
            self.params = []
            self.generate_valid_parameters(self.num_maps_to_generate)

        self.d_K0 = d_K0
        self.d_xc = d_xc
        self.d_yc = d_yc
        self.d_alpha = d_alpha
        self.d_beta = d_beta
        self.d_gamma = d_gamma
        self.d_sx = d_sx
        self.d_sy = d_sy
        self.d_cx = d_cx
        self.d_cy = d_cy

    def check_edge_point(self, x, y):
        # If edge point is inside the image by too much, then its likely the full module is not fully visible
        return (x > self.edge_tolerance
                and x < self.width - self.edge_tolerance
                and y > self.edge_tolerance
                and y < self.height - self.edge_tolerance)

    def check_valid_map(self, map_x, map_y, skip=10):
        # A higher skip means this step will be faster
        # Checking that the edges of the distorted image are either mapped to outside or on the edge of the original image
        # Top edge
        x = 0
        for y in range(0, self.width - skip, skip):
            if self.check_edge_point(map_x[x][y], map_y[x][y]):
                return False
        # Bottom edge
        x = self.height - 1
        for y in range(0, self.width - skip, skip):
            if self.check_edge_point(map_x[x][y], map_y[x][y]):
                return False
        # Left edge
        y = 0
        for x in range(0, self.height - skip, skip):
            if self.check_edge_point(map_x[x][y], map_y[x][y]):
                return False
        # Right edge
        y = self.width - 1
        for x in range(0, self.height - skip, skip):
            if self.check_edge_point(map_x[x][y], map_y[x][y]):
                return False

        # For good measure, we'll check the bottom right corner as well since it could have been skipped
        # Since it's the last one, at this point let's just return the inverse of this result
        return not self.check_edge_point(map_x[self.height - 1][y], map_y[self.height - 1][y])

    def get_rectification_transform_mtx(self, alpha, beta, gamma, sx, sy, cx, cy, cz):
        # Convert the parameters into a transformation matrix
        R_alpha = np.array([[1, 0, 0],
                            [0, np.cos(alpha), -np.sin(alpha)],
                            [0, np.sin(alpha), np.cos(alpha)]], dtype=np.float32)

        R_beta = np.array([[np.cos(beta), 0, np.sin(beta)],
                           [0, 1, 0],
                           [-np.sin(beta), 0, np.cos(beta)]], dtype=np.float32)

        R_gamma = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                            [np.sin(gamma), np.cos(gamma), 0],
                            [0, 0, 1]], dtype=np.float32)
        R = np.matmul(R_alpha, R_beta)
        R = np.matmul(R, R_gamma)

        S = [[sx, 0, 0],
             [0, sy, 0],
             [0, 0, 1]]

        t = np.array([[cx], [cy], [cz]], dtype=np.float32)

        T = np.concatenate((np.matmul(R, S), t), axis=1)
        T = np.concatenate((T, np.zeros((1, T.shape[1]))), axis=0)
        T[T.shape[0]-1, T.shape[1]-1] = 1

        return T

    def generate_random_parameters(self):
        # Gonna roll with normal distribution, open to changes
        # Only 10 parameters vary here, cz is always 0 in this experiment, so the labels will only be 8
        K = np.array([np.random.normal(0, self.d_K0),
                      0,
                      0,
                      0], dtype=np.float32)
        xc = np.random.normal(0, self.d_xc)
        yc = np.random.normal(0, self.d_yc)
        alpha = np.random.normal(0, self.d_alpha)
        beta = np.random.normal(0, self.d_beta)
        gamma = np.random.normal(0, self.d_gamma)
        sx = np.random.normal(1, self.d_sx)
        sy = np.random.normal(1, self.d_sy)
        cx = np.random.normal(0, self.d_cx)
        cy = np.random.normal(0, self.d_cy)
        cz = 0
        return K, xc, yc, alpha, beta, gamma, sx, sy, cx, cy, cz

    def standardize_numpy_parameters(self, K0, xc, yc, alpha, beta, gamma, sx, sy, cx, cy):
        return np.array([K0/self.d_K0,
                         xc/self.d_xc,
                         yc/self.d_yc,
                         alpha/self.d_alpha,
                         beta/self.d_beta,
                         gamma/self.d_gamma,
                         (sx - 1)/self.d_sx,
                         (sy - 1)/self.d_sy,
                         cx/self.d_cx,
                         cy/self.d_cy], dtype=np.float32)

    def generate_valid_parameters(self, num_to_generate: int):
        if num_to_generate > 0:
            print('Generating {} valid parameters'.format(num_to_generate))

        for i in tqdm(range(num_to_generate)):
            valid = False
            while (valid == False):
                K, xc, yc, alpha, beta, gamma, sx, sy, cx, cy, cz = self.generate_random_parameters()
                transform_matrix = self.get_rectification_transform_mtx(
                    alpha, beta, gamma, sx, sy, cx, cy, cz)
                map_x, map_y = cv2.initUndistortRectifyMap(self.cam_matrix,
                                                           K,
                                                           transform_matrix[:3, :3],
                                                           None,
                                                           (self.width,
                                                            self.height),
                                                           cv2.CV_32FC1)
                valid = self.check_valid_map(map_x, map_y)
            self.maps_x.append(map_x)
            self.maps_y.append(map_y)
            self.params.append(self.standardize_numpy_parameters(
                K[0], xc, yc, alpha, beta, gamma, sx, sy, cx, cy))

    # def add_noise_and_rubbish():
    #     # TODO:
    #     # cv2.add(img, noise)
    #     pass

    def __len__(self):
        return self.num_maps_to_generate / 20

    def __iter__(self):
        inputs = []
        labels = []
        for i in range(self.num_maps_to_generate):
            if self.use_pre_generated_maps:
                sample_module = next(self.module_generator)
                input = cv2.remap(
                    sample_module, self.maps_x[i], self.maps_y[i], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                label = self.params[i]
            else:
                input, label = self.__next__()
            labels.append(label)
            inputs.append(input.transpose())
            if i % 20 == 19:
                labels = from_numpy(np.array(labels))
                inputs = from_numpy(np.array(inputs))
                yield inputs, labels
                inputs = []
                labels = []

    def __next__(self):
        valid = False
        while (valid == False):
            K, xc, yc, alpha, beta, gamma, sx, sy, cx, cy, cz = self.generate_random_parameters()
            transform_matrix = self.get_rectification_transform_mtx(
                alpha, beta, gamma, sx, sy, cx, cy, cz)
            map_x, map_y = cv2.initUndistortRectifyMap(self.cam_matrix,
                                                       K,
                                                       transform_matrix[:3, :3],
                                                       None,
                                                       (self.width, self.height),
                                                       cv2.CV_32FC1)
            valid = self.check_valid_map(map_x, map_y)
        sample_module = next(self.module_generator)
        return cv2.remap(sample_module, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT),\
            self.standardize_numpy_parameters(
                K[0], xc, yc, alpha, beta, gamma, sx, sy, cx, cy)
