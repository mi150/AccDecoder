import configparser
import cv2
import imutils
import numpy as np
from skimage import feature
from DRL_con.video_processor import VideoProcessor
from os import path
import datetime
class DiffProcessor:
    #初始化
    def __init__(self, thresh=0, fraction=.0, dataset=None):
        """
        :param thresh: threshold, frame with diff above which will be sent
        :param fraction: only support first and second, force the fraction
        :param dataset: for loading external config
        """
        self.feature = 'none'
        self.fraction = fraction
        self.thresh = thresh
        self.section = self.get_section(dataset)
        self.name = f'{self.feature}-{self.thresh}-{self.fraction}'
    def get_res_diff_vector(self, video_path):
        diff_values = []
        prev_frame = cv2.imread(video_path + 'res' + str(1) + '.jpg')
        #cv2.imshow('origin image', prev_frame)
        prev_frame = self.get_frame_feature(prev_frame)
        i = 1
        for frame_id in range(1,2997):
            frame=cv2.imread(video_path + 'res' + str(frame_id) + '.jpg')
            frame = self.get_frame_feature(frame)
            diff_value = self.cal_frame_diff(frame, prev_frame)
            diff_values.append(diff_value)
            prev_frame = frame
            i += 1
        return diff_values
    def get_all_diff_vector(self, video_path,goplen):
        diff_values = []
        diff_valuess=[]
        diff_gop=[]
        with VideoProcessor(video_path,0) as video:
            prev_frame = next(video)
            prev_frame = self.get_frame_feature(prev_frame)
            i=1
            for frame in video:
                frame = self.get_frame_feature(frame)
                diff_value = self.cal_frame_diff(frame, prev_frame)
                if i%goplen==0:
                    diff_gop.append(diff_value)
                    diff_valuess.append(diff_values)
                    diff_values=[]
                else:
                    diff_values.append(diff_value)
                prev_frame = frame
                i+=1
        #print(diff_values)
        return diff_valuess,diff_gop

    #获取到差值的向量
    def get_diff_vector(self, video_path,index,goplen):
        diff_values = []
        with VideoProcessor(video_path,index) as video:
            prev_frame = next(video)
            prev_frame = self.get_frame_feature(prev_frame)
            i=1
            for frame in video:

                frame = self.get_frame_feature(frame)

                diff_value = self.cal_frame_diff(frame, prev_frame)
                if i==goplen:
                    return diff_values
                else:
                    diff_values.append(diff_value)
                prev_frame = frame
                i+=1
        print(diff_values)
        return diff_values
    #选择需要过滤的帧
    def process_video(self, video_path,idx):
        selected_frames = [idx]

        with VideoProcessor(video_path) as video:

            prev_frame = next(video)
            prev_feat = self.get_frame_feature(prev_frame)
            for frame in video:
                feat = self.get_frame_feature(frame)
                dis = self.cal_frame_diff(feat, prev_feat)
                if dis > self.thresh:
                    selected_frames.append(video.index+idx)
                    prev_feat = feat


        result = {
            'feature': self.feature,
            'thresh': self.thresh,
            'selected_frames': selected_frames,
            'num_selected_frames': len(selected_frames)
        }
        return result

    @staticmethod
    def batch_diff(diff_value, diff_processors):
        diff_integral = np.cumsum([0.0] + diff_value).tolist()
        diff_results = {}
        total_frames = 1 + len(diff_value)
        for dp in diff_processors:
            threshold = dp.thresh
            selected_frames = [1]
            estimations = [1.0]
            last, current = 1, 2
            while current < total_frames:
                diff_delta = diff_integral[current] - diff_integral[last]
                if diff_delta >= threshold:
                    selected_frames.append(current)
                    last = current
                    estimations.append(1.0)
                else:
                    estimations.append((threshold - diff_delta) / threshold)
                current += 1
            diff_results[dp.name] = DiffProcessor._format_result(selected_frames, total_frames, estimations)
        return diff_results

    @staticmethod
    def batch_diff_noobj_last(thresh, diff_value, index, goplen):
        a = [0.0]
        a.extend(diff_value)
        diff_integral = np.cumsum(a).tolist()
        if diff_value[0]==0:
            selected_frames = [index]
        else:
            selected_frames = []
        last, current = 0, 1
        while current < goplen:
            diff_delta = diff_integral[current] - diff_integral[last]
            if diff_delta >= thresh:
                selected_frames.append(current + index-1)
                last = current
            current += 1
        return selected_frames

    @staticmethod
    def batch_diff_noobj(thresh,diff_value,index,goplen):
        a=[0.0]
        a.extend(diff_value)
        diff_integral = np.cumsum(a).tolist()
        selected_frames = [index]
        last, current = 0, 1
        while current < goplen:
            diff_delta = diff_integral[current] - diff_integral[last]
            if diff_delta >= thresh:

                selected_frames.append(current+index)
                last = current
            current += 1
        return selected_frames

    def cal_frame_diff(self, frame, prev_frame):
        """Calculate the different between frames."""
        raise NotImplementedError()

    def get_frame_feature(self, frame):
        """Extract feature of frame."""
        raise NotImplementedError()

    @staticmethod
    def get_section(dataset):
        config = configparser.ConfigParser()

        #log_path = "config\diff_config.ini"
        #log_file_path =path.join(path.dirname(path.abspath(__file__)), log_path)
        #log_file_path = path.join(path.dirname(path.abspath(__file__)), 'config/diff_config.ini')
        log_file_path='D:\\VASRL\\config\\diff_config.ini'
        config.read(log_file_path)
        return config[dataset if dataset and dataset in config else 'default']

    def _load_section(self, section):
        return

    def __str__(self):
        return self.name

    @staticmethod
    def _format_result(selected_frames, total_frames, estimations):
        return {
            # 'fps': total_frames / complete_time if complete_time != 0 else -1,
            'selected_frames': selected_frames,
            'num_selected_frames': len(selected_frames),
            'num_total_frames': total_frames,
            'fraction': len(selected_frames) / total_frames,
            'estimation': sum(estimations) / len(estimations)
        }

    @staticmethod
    def str2class(feature):
        return {
            'pixel': PixelDiff,
            'area': AreaDiff,
            'edge': EdgeDiff,
            'corner': CornerDiff,
            'hist': HistDiff,
            'hog': HOGDiff,
            'sift': SIFTDiff,
            'surf': SURFDiff,
        }[feature]


class PixelDiff(DiffProcessor):

    feature = 'pixel'

    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        super().__init__(thresh, fraction, dataset)
        self.name = f'{self.feature}-{self.thresh}-{self.fraction}'
        self._load_section(self.section)

    def get_frame_feature(self, frame):
        return frame

    def cal_frame_diff(self, frame, prev_frame):
        total_pixels = frame.shape[0] * frame.shape[1]
        # 将两幅图像做差
        frame_diff = cv2.absdiff(frame, prev_frame)
        # 将差分后的图像转换为灰度图
        frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        # 将中灰度值小于self.pixel_thresh_low_bound的点置0，灰度值大于self.pixel_thresh_low_bound的点置255
        frame_diff = cv2.threshold(frame_diff, self.pixel_thresh_low_bound,
                                   255, cv2.THRESH_BINARY)[1]
        #得到非零的像素点数
        changed_pixels = cv2.countNonZero(frame_diff)
        #计算两帧间改变的像素个数
        fraction_changed = changed_pixels / total_pixels
        return fraction_changed
    #从配置文件中获取PIXEL_THRESH_LOW_BOUND
    def _load_section(self, section):
        self.pixel_thresh_low_bound = section.getint('PIXEL_THRESH_LOW_BOUND', 21)

#提取area特征
class AreaDiff(DiffProcessor):

    feature = 'area'
    #初始化，读取section
    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        super().__init__(thresh, fraction, dataset)
        self.name = f'{self.feature}-{self.thresh}-{self.fraction}'
        self._load_section(self.section)

    def get_frame_feature(self, frame):
        #获取灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #高斯滤波，进行图像平滑处理
        blur = cv2.GaussianBlur(gray, (self.area_blur_rad, self.area_blur_rad),
                                self.area_blur_var)
        return blur
    #对两帧求area差值
    def cal_frame_diff(self, frame, prev_frame):
        total_pixels = frame.shape[0] * frame.shape[1]

        frame_delta = cv2.absdiff(frame, prev_frame)
        thresh = cv2.threshold(frame_delta, self.area_thresh_low_bound, 255,
                               cv2.THRESH_BINARY)[1]
        ###
        #膨胀
        thresh = cv2.dilate(thresh, None)
        #寻找图像中的轮廓
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        if not contours:
            return 0.0
        #返回最大面积的轮廓，即两帧之间变化的最大面积
        return max([cv2.contourArea(c) / total_pixels for c in contours])
    #获取config中各个参数
    def _load_section(self, section):
        self.area_blur_rad = section.getint('AREA_BLUR_RAD', 11)
        self.area_blur_var = section.getint('EDGE_BLUR_VAR', 0)
        self.area_thresh_low_bound = section.getint('AREA_THRESH_LOW_BOUND', 21)

#求edge特征
class EdgeDiff(DiffProcessor):

    feature = 'edge'

    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        super().__init__(thresh, fraction, dataset)
        self.name = f'{self.feature}-{self.thresh}-{self.fraction}'
        self._load_section(self.section)

    def get_frame_feature(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self.edge_blur_rad, self.edge_blur_rad),
                                self.edge_blur_var)
        #边缘检测
        # gray_lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
        # dst = cv2.convertScaleAbs(gray_lap)
        edge = cv2.Canny(blur, self.edge_canny_low,  self.edge_canny_high)
        return edge
    #对边缘进行差值计算
    def cal_frame_diff(self, edge, prev_edge):
        total_pixels = edge.shape[0] * edge.shape[1]
        frame_diff = cv2.absdiff(edge, prev_edge)
        frame_diff = cv2.threshold(frame_diff, self.edge_thresh_low_bound, 255,
                                   cv2.THRESH_BINARY)[1]
        changed_pixels = cv2.countNonZero(frame_diff)
        fraction_changed = changed_pixels / total_pixels
        return fraction_changed
    #取参数
    def _load_section(self, section):
        self.edge_blur_rad = section.getint('EDGE_BLUR_RAD', 5)
        self.edge_blur_var = section.getint('EDGE_BLUR_VAR', 0)
        self.edge_canny_low = section.getint('EDGE_CANNY_LOW', 101)
        self.edge_canny_high = section.getint('EDGE_CANNY_HIGH', 255)
        self.edge_thresh_low_bound = section.getint('EDGE_THRESH_LOW_BOUND', 21)

#计算corner特征
class CornerDiff(DiffProcessor):

    feature = 'corner'

    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        super().__init__(thresh, fraction, dataset)
        self.name = f'{self.feature}-{self.thresh}-{self.fraction}'
        self._load_section(self.section)
    #conrner角点检测
    def get_frame_feature(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corner = cv2.cornerHarris(gray, self.corner_block_size,
                                  self.corner_ksize, self.corner_k)
        corner = cv2.dilate(corner, None)
        return corner
    #求差
    def cal_frame_diff(self, corner, prev_corner):
        total_pixels = corner.shape[0] * corner.shape[1]
        frame_diff = cv2.absdiff(corner, prev_corner)
        changed_pixels = cv2.countNonZero(frame_diff)
        fraction_changed = changed_pixels / total_pixels
        return fraction_changed

    def _load_section(self, section):
        self.corner_block_size = section.getint('CORNER_BLOCK_SIZE', 5)
        self.corner_ksize = section.getint('CORNER_KSIZE', 3)
        self.corner_k = section.getfloat('CORNER_K', 0.05)

#直方图特征
class HistDiff(DiffProcessor):

    feature = 'histogram'

    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        super().__init__(thresh, fraction, dataset)
        self.name = f'{self.feature}-{self.thresh}-{self.fraction}'
    #获取直方图
    def get_frame_feature(self, frame):
        nb_channels = frame.shape[-1]
        hist = np.zeros((self.hist_nb_bins * nb_channels, 1), dtype='float32')
        for i in range(nb_channels):
            hist[i * self.hist_nb_bins: (i + 1) * self.hist_nb_bins] = \
                cv2.calcHist(frame, [i], None, [self.hist_nb_bins], [0, 256])
        hist = cv2.normalize(hist, hist)
        return hist

    #直方图的卡方比较
    def cal_frame_diff(self, frame, prev_frame):
        return cv2.compareHist(frame, prev_frame, cv2.HISTCMP_CHISQR)

    def _load_section(self, section):
        self.hist_nb_bins = section.getint('HIST_NB_BINS', 32)

#方向梯度直方图HOG
class HOGDiff(DiffProcessor):

    feature = 'HOG'

    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        super().__init__(thresh, fraction, dataset)
        self.name = f'{self.feature}-{self.thresh}-{self.fraction}'
        self._load_section(self.section)

    def get_frame_feature(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize to speed up
        gray = cv2.resize(gray, (self.hog_resize, self.hog_resize))
        #求梯度直方图
        hog = feature.hog(gray, orientations=self.hog_orientations,
                          pixels_per_cell=(self.hog_pixel_cell, self.hog_pixel_cell),
                          cells_per_block=(self.hog_cell_block, self.hog_cell_block)
                          ).astype('float32')
        return hog

    def cal_frame_diff(self, frame, prev_frame):
       #是求整体的矩阵元素平方和，再开根号。求范数
        dis = np.linalg.norm(frame - prev_frame)
        dis /= frame.shape[0]
        return dis

    def _load_section(self, section):
        self.hog_resize = section.getint('HOG_RESIZE', 128)
        self.hog_orientations = section.getint('HOG_ORIENTATIONS', 10)
        self.hog_pixel_cell = section.getint('HOG_PIXEL_CELL', 7)
        self.hog_cell_block = section.getint('HOG_CELL_BLOCK', 3)

#求sift
class SIFTDiff(DiffProcessor):

    feature = 'SIFT'

    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        super().__init__(thresh, fraction, dataset)
        self.name = f'{self.feature}-{self.thresh}-{self.fraction}'

    def get_frame_feature(self, frame):
        #实例化sift函数
        sift = cv2.xfeatures2d.SIFT_create()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #求描述符
        _, des = sift.detectAndCompute(gray, None)
        #压缩行
        des = (np.mean(des, axis=0).astype('float32')
               if des is not None else np.zeros(128))
        return des

    def cal_frame_diff(self, frame, prev_frame):
        dis = np.linalg.norm(frame - prev_frame)
        dis /= frame.shape[0]
        return dis

#SURF
class SURFDiff(DiffProcessor):

    feature = 'SURF'

    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        super().__init__(thresh, fraction, dataset)
        self._load_section(self.section)
        self.name = f'{self.feature}-{self.thresh}-{self.fraction}'

    def get_frame_feature(self, frame):
        surf = cv2.xfeatures2d.SURF_create()
        #要检测方向
        surf.setUpright(True)
        #设置Hessian矩阵的阈值
        surf.setHessianThreshold(self.surf_hessian_thresh)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, des = surf.detectAndCompute(gray, None)
        des = np.zeros(128) if des is None else np.mean(des, axis=0).astype('float32')
        return des

    def cal_frame_diff(self, frame, prev_frame):
        dis = np.linalg.norm(frame - prev_frame)
        dis /= frame.shape[0]
        return dis

    def _load_section(self, section):
        self.surf_hessian_thresh = section.getint('SURF_HESSIAN_THRESH', 400)