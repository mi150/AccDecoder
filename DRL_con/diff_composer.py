#from reducto.data_loader import load_json
import diff_processor


class DiffComposer:

    def __init__(self, differ_dict=None):
        self.differ_dict = differ_dict

    # @staticmethod
    # def from_jsonfile(jsonpath, differencer_types=None):
    #     differencer_types = differencer_types or ['pixel', 'area', 'corner', 'edge']
    #     differ_dict = load_json(jsonpath)
    #     differencers = {
    #         feature: threshes
    #         for feature, threshes in differ_dict.items()
    #         if feature in differencer_types
    #     }
    #     return DiffComposer(differencers)

    @staticmethod
    def placeholder(differencer_types=None):
        differencer_types = differencer_types or ['pixel', 'area', 'corner', 'edge']
        differencers = {
            feature: 0
            for feature in differencer_types
        }
        return DiffComposer(differencers)

    def new_thresholds(self, thresholds):
        for dp, threshes in thresholds.items():
            self.differ_dict[dp] = threshes

    def process_video(self, filepath, diff_vectors=None):
        if diff_vectors:
            assert all([k in diff_vectors for k in self.differ_dict.keys()]), \
                'not compatible diff-vector list'
        else:
            diff_vectors = {
                k: self.get_diff_vector(k, filepath)
                for k in self.differ_dict.keys()
            }

        results = {}
        for differ_type, thresholds in self.differ_dict.items():
            diff_vector = diff_vectors[differ_type]
            result = self.batch_diff(diff_vector, thresholds)
            results[differ_type] = {
                'diff_vector': diff_vector,
                'result': result,
            }
        return results

    @staticmethod
    def get_diff_vector(differ_type, filepath):
        differ = diff_processor.DiffProcessor.str2class(differ_type)()
        vectors=differ.get_diff_vector(filepath)
        diff_value_range = None
        diffmin=[]
        diffmax=[]
        for vector in vectors:
            # print(vector)
            diffmin.append(min(vector))
            diffmax.append(max(vector))
            if diff_value_range is None:
                diff_value_range = (min(vector), max(vector))
            else:
                diff_value_range = (min([min(vector), diff_value_range[0]]),
                                    max([max(vector), diff_value_range[1]]))
        print(diffmin)
        print(diffmax)
        return vectors,diff_value_range

    @staticmethod
    def batch_diff(diff_vector, thresholds):
        result = diff_processor.DiffProcessor.batch_diff_noobj(diff_vector, thresholds)
        return result